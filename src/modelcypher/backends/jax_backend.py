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

"""JAX Backend for ModelCypher.

Provides hardware-invariant tensor operations using JAX, enabling:
- TPU support for large-scale manifold analysis
- GPU acceleration via CUDA/ROCm
- Composable transformations (jit, vmap, grad) for geometry operations

JAX is well-suited for high-dimensional geometry work due to:
- XLA compilation for efficient tensor operations
- Automatic differentiation for Jacobian/Hessian computation
- Functional purity enabling reproducible research
"""

from __future__ import annotations

from typing import Any, Callable

from modelcypher.backends.conversion_utils import (
    raise_numpy_disabled,
    to_list_with_eval,
    to_scalar_with_eval,
)
from modelcypher.ports.backend import Array, Backend, FloatInfo


class JAXBackend(Backend):
    """JAX implementation of the Backend protocol.

    Uses jax.numpy for array operations and jax.scipy.linalg for
    linear algebra. Random operations use explicit PRNG keys for
    reproducibility.
    """

    def __init__(self) -> None:
        import jax
        import jax.numpy as jnp

        self.jax = jax
        self.jnp = jnp
        self._rng_key = jax.random.PRNGKey(0)
        self._compiled_cache: dict[str, Callable] = {}

    def _next_key(self) -> Any:
        """Get next PRNG key and update internal state.

        Returns
        -------
        Any
            PRNG key for random operations.
        """
        # Split key to maintain functional purity
        self._rng_key, subkey = self.jax.random.split(self._rng_key)
        return subkey

    # --- Array Creation ---
    def array(self, data: Any, dtype: Any | None = None) -> Array:
        return self.jnp.array(data, dtype=self._map_dtype(dtype))

    def zeros(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        return self.jnp.zeros(shape, dtype=self._map_dtype(dtype))

    def ones(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        return self.jnp.ones(shape, dtype=self._map_dtype(dtype))

    def eye(self, n: int, m: int | None = None, dtype: Any | None = None) -> Array:
        return self.jnp.eye(n, m, dtype=self._map_dtype(dtype))

    def arange(
        self,
        start: int | float,
        stop: int | float | None = None,
        step: int | float = 1,
        dtype: Any | None = None,
    ) -> Array:
        if stop is None:
            return self.jnp.arange(start, dtype=self._map_dtype(dtype))
        return self.jnp.arange(start, stop, step, dtype=self._map_dtype(dtype))

    def triu_indices(self, n: int, k: int = 0) -> tuple[Array, Array]:
        return self.jnp.triu_indices(n, k=k)

    def diag(self, array: Array, k: int = 0) -> Array:
        return self.jnp.diag(array, k=k)

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: Any | None = None) -> Array:
        return self.jnp.full(shape, fill_value, dtype=self._map_dtype(dtype))

    def dtype(self, array: Array) -> Any:
        """Return the dtype of an array."""
        return array.dtype

    def ones_like(self, array: Array, dtype: Any | None = None) -> Array:
        return self.jnp.ones_like(array, dtype=self._map_dtype(dtype))

    def zeros_like(self, array: Array, dtype: Any | None = None) -> Array:
        return self.jnp.zeros_like(array, dtype=self._map_dtype(dtype))

    def linspace(self, start: float, stop: float, num: int, dtype: Any | None = None) -> Array:
        return self.jnp.linspace(start, stop, num, dtype=self._map_dtype(dtype))

    def meshgrid(self, *arrays: Array, indexing: str = "xy") -> list[Array]:
        return list(self.jnp.meshgrid(*arrays, indexing=indexing))

    # --- Shape Manipulation ---
    def shape(self, array: Array) -> tuple[int, ...]:
        return tuple(array.shape)

    def reshape(self, array: Array, shape: tuple[int, ...]) -> Array:
        return self.jnp.reshape(array, shape)

    def squeeze(self, array: Array, axis: int | None = None) -> Array:
        return self.jnp.squeeze(array, axis=axis)

    def transpose(self, array: Array, axes: tuple[int, ...] | None = None) -> Array:
        return self.jnp.transpose(array, axes=axes)

    def stack(self, arrays: list[Array], axis: int = 0) -> Array:
        return self.jnp.stack(arrays, axis=axis)

    def concatenate(self, arrays: list[Array], axis: int = 0) -> Array:
        return self.jnp.concatenate(arrays, axis=axis)

    def broadcast_to(self, array: Array, shape: tuple[int, ...]) -> Array:
        return self.jnp.broadcast_to(array, shape)

    def tile(self, array: Array, reps: tuple[int, ...] | int) -> Array:
        return self.jnp.tile(array, reps)

    def expand_dims(self, array: Array, axis: int | tuple[int, ...]) -> Array:
        return self.jnp.expand_dims(array, axis=axis)

    # --- Reductions ---
    def sum(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.jnp.sum(array, axis=axis, keepdims=keepdims)

    def mean(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.jnp.mean(array, axis=axis, keepdims=keepdims)

    def max(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return self.jnp.max(array, axis=axis, keepdims=keepdims)

    def min(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return self.jnp.min(array, axis=axis, keepdims=keepdims)

    def argmax(self, array: Array, axis: int | None = None) -> Array:
        return self.jnp.argmax(array, axis=axis)

    def argmin(self, array: Array, axis: int | None = None) -> Array:
        return self.jnp.argmin(array, axis=axis)

    def var(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.jnp.var(array, axis=axis, keepdims=keepdims)

    def std(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.jnp.std(array, axis=axis, keepdims=keepdims)

    def all(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.jnp.all(array, axis=axis, keepdims=keepdims)

    def any(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.jnp.any(array, axis=axis, keepdims=keepdims)

    # --- Element-wise Operations ---
    def sqrt(self, array: Array) -> Array:
        return self.jnp.sqrt(array)

    def exp(self, array: Array) -> Array:
        return self.jnp.exp(array)

    def log(self, array: Array) -> Array:
        return self.jnp.log(array)

    def abs(self, array: Array) -> Array:
        return self.jnp.abs(array)

    def sign(self, array: Array) -> Array:
        return self.jnp.sign(array)

    def isnan(self, array: Array) -> Array:
        return self.jnp.isnan(array)

    def isinf(self, array: Array) -> Array:
        return self.jnp.isinf(array)

    def isfinite(self, array: Array) -> Array:
        return self.jnp.isfinite(array)

    def sin(self, array: Array) -> Array:
        return self.jnp.sin(array)

    def cos(self, array: Array) -> Array:
        return self.jnp.cos(array)

    def arccos(self, array: Array) -> Array:
        return self.jnp.arccos(array)

    def arctan(self, array: Array) -> Array:
        return self.jnp.arctan(array)

    def lgamma(self, array: Array) -> Array:
        return self.jax.scipy.special.gammaln(array)

    def maximum(self, lhs: Array, rhs: Array) -> Array:
        return self.jnp.maximum(lhs, rhs)

    def minimum(self, lhs: Array, rhs: Array) -> Array:
        return self.jnp.minimum(lhs, rhs)

    def add(self, lhs: Array | float, rhs: Array | float) -> Array:
        """Element-wise addition."""
        return self.jnp.add(lhs, rhs)

    def subtract(self, lhs: Array | float, rhs: Array | float) -> Array:
        """Element-wise subtraction."""
        return self.jnp.subtract(lhs, rhs)

    def multiply(self, lhs: Array | float, rhs: Array | float) -> Array:
        """Element-wise multiplication."""
        return self.jnp.multiply(lhs, rhs)

    def divide(self, lhs: Array | float, rhs: Array | float) -> Array:
        """Element-wise division."""
        return self.jnp.divide(lhs, rhs)

    def clip(
        self, array: Array, min_val: float | Array | None, max_val: float | Array | None
    ) -> Array:
        return self.jnp.clip(array, min_val, max_val)

    def where(self, condition: Array, x: Array, y: Array) -> Array:
        return self.jnp.where(condition, x, y)

    def softmax(self, array: Array, axis: int = -1) -> Array:
        from jax.nn import softmax

        return softmax(array, axis=axis)

    def cumsum(self, array: Array, axis: int | None = None) -> Array:
        return self.jnp.cumsum(array, axis=axis)

    def floor(self, array: Array) -> Array:
        return self.jnp.floor(array)

    def ceil(self, array: Array) -> Array:
        return self.jnp.ceil(array)

    def log2(self, array: Array) -> Array:
        return self.jnp.log2(array)

    def mod(self, lhs: Array, rhs: Array | float | int) -> Array:
        return self.jnp.mod(lhs, rhs)

    # --- Linear Algebra ---
    def matmul(self, lhs: Array, rhs: Array) -> Array:
        return self.jnp.matmul(lhs, rhs)

    def dot(self, a: Array, b: Array) -> Array:
        return self.jnp.dot(a, b)

    def svd(self, array: Array, compute_uv: bool = True) -> tuple[Array, Array, Array] | Array:
        if compute_uv:
            u, s, vt = self.jnp.linalg.svd(array, full_matrices=False)
            return u, s, vt
        return self.jnp.linalg.svd(array, compute_uv=False)

    def norm(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.jnp.linalg.norm(array, axis=axis, keepdims=keepdims)

    def det(self, array: Array) -> Array:
        return self.jnp.linalg.det(array)

    def linalg_det(self, array: Array) -> Array:
        """Alias for det() for compatibility."""
        return self.det(array)

    def eigh(self, array: Array) -> tuple[Array, Array]:
        eigenvalues, eigenvectors = self.jnp.linalg.eigh(array)
        return eigenvalues, eigenvectors

    def eigvalsh(self, array: Array) -> Array:
        """Compute eigenvalues of symmetric/Hermitian matrix (values only, more efficient)."""
        return self.jnp.linalg.eigvalsh(array)

    def solve(self, a: Array, b: Array) -> Array:
        return self.jnp.linalg.solve(a, b)

    def inv(self, array: Array) -> Array:
        return self.jnp.linalg.inv(array)

    def pinv(self, array: Array) -> Array:
        return self.jnp.linalg.pinv(array)

    def cholesky(self, array: Array) -> Array:
        return self.jnp.linalg.cholesky(array)

    def trace(self, array: Array) -> Array:
        return self.jnp.trace(array)

    def qr(self, array: Array) -> tuple[Array, Array]:
        q, r = self.jnp.linalg.qr(array)
        return q, r

    def matrix_sqrt_newton_schulz(self, A: Array, num_iters: int = 15) -> Array:
        """Compute matrix square root via Newton-Schulz iteration.

        Converges to A^{1/2} for positive semi-definite A.
        Runs entirely on GPU/TPU.

        Algorithm:
            Y₀ = A / norm(A)
            Z₀ = I
            for k=0..N:
                T = (3I - ZₖYₖ)
                Yₖ₊₁ = ½ Yₖ T
                Zₖ₊₁ = ½ T Zₖ

            A^{1/2} ≈ Y_final * sqrt(norm(A))
        """
        # Ensure float32 for stability
        A_f32 = A.astype(self.jnp.float32)

        # Scaling to ensure convergence (spectral norm <= 1)
        padding = self.jnp.array(1e-7, dtype=self.jnp.float32)
        normA_val = self.jnp.sqrt(self.jnp.sum(A_f32 * A_f32)) + padding
        Y = A_f32 / normA_val

        shape = A.shape
        I = self.jnp.eye(shape[0], dtype=self.jnp.float32)
        Z = I

        three = self.jnp.array(3.0, dtype=self.jnp.float32)
        half = self.jnp.array(0.5, dtype=self.jnp.float32)

        # Iteration
        for _ in range(num_iters):
            ZY = self.jnp.matmul(Z, Y)
            T = three * I - ZY

            Y_new = half * self.jnp.matmul(Y, T)
            Z_new = half * self.jnp.matmul(T, Z)

            Y = Y_new
            Z = Z_new

        # Rescale
        sqrtA = Y * self.jnp.sqrt(normA_val)

        # Cast back if necessary
        if A.dtype != self.jnp.float32:
            sqrtA = sqrtA.astype(A.dtype)

        return sqrtA

    def floyd_warshall(self, dist: Array) -> Array:
        """Compute all-pairs shortest paths using Floyd-Warshall on device."""
        mat = self.jnp.array(dist)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError("floyd_warshall requires a square [n, n] matrix")
        n = int(mat.shape[0])
        if n <= 1:
            return mat

        cache_key = f"floyd_warshall_{n}_{mat.dtype}"
        compiled = self._compiled_cache.get(cache_key)
        if compiled is None:
            def _fw(d: Array) -> Array:
                def body(k, current):
                    via = current[:, k : k + 1] + current[k : k + 1, :]
                    return self.jnp.minimum(current, via)

                return self.jax.lax.fori_loop(0, n, body, d)

            compiled = self.jax.jit(_fw)
            self._compiled_cache[cache_key] = compiled

        return compiled(mat)

    def single_source_shortest_paths(self, dist: Array, source_index: int) -> Array:
        """Compute shortest paths from a single source using Dijkstra-style relaxation."""
        mat = self.jnp.array(dist)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError("single_source_shortest_paths requires a square [n, n] matrix")
        n = int(mat.shape[0])
        if n <= 1:
            return mat[0] if n == 1 else mat

        src = int(source_index)
        if src < 0 or src >= n:
            raise ValueError("source_index out of bounds")

        cache_key = f"sssp_{n}_{mat.dtype}_{src}"
        compiled = self._compiled_cache.get(cache_key)
        if compiled is None:
            def _sssp(d: Array) -> Array:
                idx = self.jnp.arange(n)
                dist_vec = d[src]
                visited = (idx == src).astype(d.dtype)
                inf_val = self.jnp.finfo(d.dtype).max

                def body(_, state):
                    dist_vec, visited = state
                    masked = dist_vec + visited * inf_val
                    min_idx = self.jnp.argmin(masked)
                    is_min = idx == min_idx
                    visited = self.jnp.minimum(visited + is_min.astype(d.dtype), 1.0)
                    row = d[min_idx]
                    dist_at_min = dist_vec[min_idx]
                    alt = dist_at_min + row
                    dist_vec = self.jnp.minimum(dist_vec, alt)
                    return dist_vec, visited

                dist_vec, _ = self.jax.lax.fori_loop(0, n - 1, body, (dist_vec, visited))
                return dist_vec

            compiled = self.jax.jit(_sssp)
            self._compiled_cache[cache_key] = compiled

        return compiled(mat)

    # --- Indexing ---
    def take(self, array: Array, indices: Array, axis: int | None = None) -> Array:
        return self.jnp.take(array, indices, axis=axis)

    def take_along_axis(self, array: Array, indices: Array, axis: int) -> Array:
        indices_int = indices.astype(self.jnp.int32)
        return self.jnp.take_along_axis(array, indices_int, axis=axis)

    def put_along_axis(
        self, array: Array, indices: Array, values: Array, axis: int | None = None
    ) -> Array:
        indices_int = indices.astype(self.jnp.int32)
        return self.jnp.put_along_axis(array, indices_int, values, axis=axis)

    # --- Sorting ---
    def sort(self, array: Array, axis: int = -1) -> Array:
        return self.jnp.sort(array, axis=axis)

    def argsort(self, array: Array, axis: int = -1) -> Array:
        return self.jnp.argsort(array, axis=axis)

    def argpartition(self, array: Array, kth: int, axis: int = -1) -> Array:
        # JAX doesn't have argpartition; use argsort as fallback
        # This is less efficient but maintains correctness
        sorted_indices = self.jnp.argsort(array, axis=axis)
        return sorted_indices

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
        return self.jnp.partition(array, kth=kth, axis=axis)

    def nonzero(self, array: Array) -> tuple[Array, ...]:
        """Find indices of non-zero elements.

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
        return self.jnp.nonzero(array)

    # --- Random ---
    def random_normal(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        key = self._next_key()
        arr = self.jax.random.normal(
            key, shape=shape, dtype=self._map_dtype(dtype) or self.jnp.float32
        )
        return arr

    def random_uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        shape: tuple[int, ...] | None = None,
        dtype: Any | None = None,
    ) -> Array:
        key = self._next_key()
        arr = self.jax.random.uniform(
            key,
            shape=shape or (1,),
            minval=low,
            maxval=high,
            dtype=self._map_dtype(dtype) or self.jnp.float32,
        )
        return arr

    def random_randint(self, low: int, high: int, shape: tuple[int, ...] | None = None) -> Array:
        key = self._next_key()
        return self.jax.random.randint(key, shape=shape or (1,), minval=low, maxval=high)

    def random_seed(self, seed: int) -> None:
        self._rng_key = self.jax.random.PRNGKey(seed)

    def random_categorical(self, logits: Array, num_samples: int = 1) -> Array:
        """Sample from categorical distribution defined by logits.

        Samples indices from a categorical distribution parameterized by
        unnormalized log-probabilities (logits).

        Args:
            logits: Array of shape (..., num_categories) containing logits.
                Can be 1D (single distribution) or 2D (batch of distributions).
            num_samples: Number of samples to draw per distribution.

        Returns:
            Array of sampled indices. Shape depends on input:
            - 1D logits: shape (num_samples,)
            - 2D logits (batch_size, num_categories): shape (batch_size, num_samples)
        """
        key = self._next_key()
        # JAX categorical expects logits of shape (..., num_classes)
        # and returns samples of shape (..., num_samples)
        return self.jax.random.categorical(key, logits, shape=(num_samples,))

    def randperm(self, n: int) -> Array:
        """Generate a random permutation of integers from 0 to n-1."""
        key = self._next_key()
        return self.jax.random.permutation(key, n)

    # --- Type Conversion ---
    def astype(self, array: Array, dtype: Any) -> Array:
        return array.astype(self._map_dtype(dtype))

    def to_numpy(self, array: Array) -> Any:
        """DISABLED: CPU arrays are not permitted in ModelCypher.

        Use backend.tolist() or backend.to_scalar() for extracting values.
        Use backend.save_safetensors() for serialization.
        """
        raise_numpy_disabled()

    def to_scalar(self, array: Array) -> float | int:
        """Extract a scalar from a 0-d or single-element array.

        Faster than to_numpy().item() - uses JAX's native .item() directly,
        skipping numpy conversion entirely.

        Args:
            array: A scalar (0-d) or single-element array.

        Returns:
            Python float or int.

        Raises:
            ValueError: If array has more than one element.
        """
        return to_scalar_with_eval(array, self.eval)

    def tolist(self, array: Array) -> list | float | int:
        """Convert array to nested Python lists.

        Uses JAX's native tolist() - MUCH faster than element-by-element to_scalar().
        """
        return to_list_with_eval(array, self.eval)

    def finfo(self, dtype: Any | None = None) -> FloatInfo:
        """Return floating-point precision info for the given dtype.

        Derives numerical stability constants from the actual dtype precision.
        """
        resolved = self._map_dtype(dtype) or self.jnp.float32
        info = self.jnp.finfo(resolved)
        return FloatInfo(
            eps=float(info.eps),
            tiny=float(info.tiny),
            max=float(info.max),
            min=float(info.min),
        )

    # --- Quantization ---
    def quantize(
        self,
        weight: Array,
        group_size: int,
        bits: int,
        mode: str,
    ) -> tuple[Array, Array, Array | None]:
        # JAX quantization - basic implementation
        # For production, consider using AQT (Accurate Quantized Training)
        shape = weight.shape
        if len(shape) < 2:
            weight = weight.reshape(-1, 1)

        num_groups = weight.shape[0] // group_size
        weight_grouped = weight.reshape(num_groups, group_size, -1)

        # Compute scales per group
        max_vals = self.jnp.max(self.jnp.abs(weight_grouped), axis=1, keepdims=True)
        scales = max_vals / (2 ** (bits - 1) - 1)
        scales = self.jnp.where(scales == 0, 1.0, scales)

        # Quantize
        weight_q = self.jnp.round(weight_grouped / scales)
        weight_q = self.jnp.clip(weight_q, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        weight_q = weight_q.astype(self.jnp.int8)

        return weight_q.reshape(shape), scales.reshape(-1, weight.shape[-1]), None

    def dequantize(
        self,
        weight: Array,
        scales: Array,
        biases: Array | None,
        group_size: int,
        bits: int,
        mode: str,
    ) -> Array:
        shape = weight.shape
        weight = weight.astype(self.jnp.float32)

        if len(shape) < 2:
            weight = weight.reshape(-1, 1)

        num_groups = weight.shape[0] // group_size
        weight_grouped = weight.reshape(num_groups, group_size, -1)
        scales_grouped = scales.reshape(num_groups, 1, -1)

        dequantized = weight_grouped * scales_grouped
        if biases is not None:
            biases_grouped = biases.reshape(num_groups, 1, -1)
            dequantized = dequantized + biases_grouped

        return dequantized.reshape(shape)

    # --- Attention Masks ---
    def create_causal_mask(self, seq_len: int, dtype: Any | None = None) -> Array:
        """Create additive causal attention mask for autoregressive models.

        Returns an upper triangular matrix filled with -inf above the diagonal,
        used to prevent attention to future tokens in autoregressive decoding.

        Args:
            seq_len: Sequence length for the square mask.
            dtype: Optional dtype for the mask (defaults to float32).

        Returns:
            A (seq_len, seq_len) tensor with 0s on/below diagonal and -inf above.
        """
        # Create lower triangular mask where future positions are -inf
        mask = self.jnp.triu(self.jnp.full((seq_len, seq_len), float("-inf")), k=1)
        if dtype is not None:
            mask = mask.astype(self._map_dtype(dtype))
        return mask

    # --- Compute Control ---
    def eval(self, *arrays: Array) -> None:
        """Force evaluation and synchronization of arrays.

        Calls block_until_ready() to ensure asynchronous XLA computation
        completes before returning.
        """
        for arr in arrays:
            if hasattr(arr, "block_until_ready"):
                arr.block_until_ready()

    def clear_cache(self) -> None:
        """Clear memory cache. JAX manages memory automatically but gc helps."""
        import gc

        gc.collect()

    def _map_dtype(self, dtype: Any | None) -> Any | None:
        if dtype is None:
            return None
        # Handle string dtype names
        if isinstance(dtype, str):
            dtype_map = {
                "float32": self.jnp.float32,
                "float64": self.jnp.float64,
                "float16": self.jnp.float16,
                "bfloat16": self.jnp.bfloat16,
                "int32": self.jnp.int32,
                "int64": self.jnp.int64,
                "int16": self.jnp.int16,
                "int8": self.jnp.int8,
                "uint8": self.jnp.uint8,
                "bool": self.jnp.bool_,
            }
            return dtype_map.get(dtype, dtype)
        # Handle dtype objects by name (covers numpy/mlx dtypes without importing them)
        name = getattr(dtype, "name", None) or getattr(dtype, "__name__", None) or str(dtype)
        name = name.replace("jax.numpy.", "")
        dtype_map = {
            "float32": self.jnp.float32,
            "float64": self.jnp.float64,
            "float16": self.jnp.float16,
            "bfloat16": self.jnp.bfloat16,
            "int32": self.jnp.int32,
            "int64": self.jnp.int64,
            "int16": self.jnp.int16,
            "int8": self.jnp.int8,
            "uint8": self.jnp.uint8,
            "bool": self.jnp.bool_,
        }
        return dtype_map.get(name, dtype)

    # =========================================================================
    # SOTA PERFORMANCE APIs (JAX)
    # =========================================================================

    def compile(
        self,
        fun: Callable,
        inputs: list | None = None,
        outputs: list | None = None,
        shapeless: bool = False,
    ) -> Callable:
        """JIT-compile a function using XLA.

        Parameters
        ----------
        fun : Callable
            Function to compile.
        inputs : list, optional
            Unused, kept for API compatibility with MLX.
        outputs : list, optional
            Unused, kept for API compatibility with MLX.
        shapeless : bool, optional
            Unused in JAX (shapes always traced). Default is False.

        Returns
        -------
        Callable
            JIT-compiled function with XLA optimizations.
        """
        return self.jax.jit(fun)

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
        return self.jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)

    def value_and_grad(self, fun: Callable, argnums: int | list[int] = 0) -> Callable:
        """Return a function that computes both value and gradient of fun.

        Parameters
        ----------
        fun : Callable
            Function to differentiate. Must return a scalar.
        argnums : int or list[int], optional
            Which positional argument(s) to differentiate with respect to.
            Default is 0.

        Returns
        -------
        Callable
            Function that returns (value, gradient) tuple.
        """
        return self.jax.value_and_grad(fun, argnums=argnums)

    def async_eval(self, *arrays: Array) -> None:
        """Asynchronously evaluate arrays without blocking.

        Parameters
        ----------
        *arrays : Array
            Arrays to evaluate asynchronously.

        Notes
        -----
        JAX operations are asynchronous by default via XLA dispatch.
        This is a no-op for API compatibility.
        """
        # JAX is async by default - XLA dispatches operations asynchronously.
        # No explicit action needed.
        return None

    # --- Fused Kernels ---

    def rms_norm(
        self,
        x: Array,
        weight: Array | None,
        eps: float = 1e-5,
        stream: Any | None = None,
    ) -> Array:
        """Apply RMS normalization.

        Parameters
        ----------
        x : Array
            Input array to normalize.
        weight : Array or None
            Scaling weights. If None, returns normalized input.
        eps : float, optional
            Epsilon for numerical stability. Default is 1e-5.
        stream : Any, optional
            Unused in JAX. Included for API compatibility.

        Returns
        -------
        Array
            RMS-normalized output.
        """
        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        variance = self.jnp.mean(x**2, axis=-1, keepdims=True)
        x_normed = x * self.jax.lax.rsqrt(variance + eps)
        if weight is None:
            return x_normed
        return x_normed * weight

    def layer_norm(
        self,
        x: Array,
        weight: Array | None,
        bias: Array | None,
        eps: float = 1e-5,
        stream: Any | None = None,
    ) -> Array:
        """Apply layer normalization.

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
            Unused in JAX. Included for API compatibility.

        Returns
        -------
        Array
            Layer-normalized output.
        """
        # LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
        mean = self.jnp.mean(x, axis=-1, keepdims=True)
        variance = self.jnp.var(x, axis=-1, keepdims=True)
        x_normed = (x - mean) * self.jax.lax.rsqrt(variance + eps)
        if weight is not None:
            x_normed = x_normed * weight
        if bias is not None:
            x_normed = x_normed + bias
        return x_normed

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
        """Apply rotary position embeddings.

        Parameters
        ----------
        x : Array
            Input array of shape (..., seq_len, dims).
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
            Unused in JAX. Included for API compatibility.

        Returns
        -------
        Array
            Output with rotary position embeddings applied.
        """
        seq_len = x.shape[-2]
        half_dims = dims // 2

        if freqs is None:
            if base is None:
                raise ValueError("rope() expects base when freqs is not provided")
            inv_freq = 1.0 / (
                base ** (self.jnp.arange(0, half_dims, dtype=x.dtype) / half_dims)
            )
            positions = self.jnp.arange(seq_len, dtype=x.dtype)
            if isinstance(offset, (int, float)):
                positions = positions + offset
            else:
                offset_arr = self.jnp.asarray(offset, dtype=x.dtype)
                positions = positions + offset_arr[..., None]
            positions = positions * scale
            freqs = positions[..., None] * inv_freq
        else:
            if base is not None:
                raise ValueError("rope() expects base=None when freqs is provided")
            freqs = freqs

        cos = self.jnp.cos(freqs)
        sin = self.jnp.sin(freqs)

        # Reshape for broadcasting
        if cos.ndim == 2:
            cos = cos.reshape((1,) * (x.ndim - 2) + cos.shape)
            sin = sin.reshape((1,) * (x.ndim - 2) + sin.shape)
        elif cos.ndim == 3:
            cos = cos.reshape((cos.shape[0],) + (1,) * (x.ndim - 3) + cos.shape[1:])
            sin = sin.reshape((sin.shape[0],) + (1,) * (x.ndim - 3) + sin.shape[1:])

        # Split x into two halves
        x1 = x[..., :half_dims]
        x2 = x[..., half_dims:dims]

        if traditional:
            # Traditional RoPE: interleaved rotation
            cos_full = self.jnp.tile(cos, (1, 1, 1, 2))
            sin_full = self.jnp.tile(sin, (1, 1, 1, 2))
            rotated = self.jnp.concatenate([-x2, x1], axis=-1)
            x_rope = x[..., :dims] * cos_full + rotated * sin_full
        else:
            # Modern RoPE: paired rotation
            x_rope = self.jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

        # Preserve dimensions beyond RoPE range
        if x.shape[-1] > dims:
            return self.jnp.concatenate([x_rope, x[..., dims:]], axis=-1)
        return x_rope

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
        """Compute scaled dot-product attention.

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
            Attention mask. Use "causal" for causal masking.
        sinks : Array or None, optional
            Attention sinks. Not supported in JAX backend.
        stream : Any, optional
            Unused in JAX. Included for API compatibility.

        Returns
        -------
        Array
            Attention output.
        """
        # Compute attention scores: Q @ K^T * scale
        scores = self.jnp.einsum("...qhd,...khd->...hqk", q, k) * scale

        # Apply mask if provided
        if mask is not None:
            if isinstance(mask, str):
                if mask != "causal":
                    raise ValueError(f"Unsupported attention mask: {mask}")
                t_q = q.shape[-2]
                t_kv = k.shape[-2]
                mask = self.jnp.tril(self.jnp.ones((t_q, t_kv), dtype=bool))
            mask_arr = self.jnp.asarray(mask)
            if mask_arr.dtype == self.jnp.bool_:
                neg_inf = self.jnp.finfo(scores.dtype).min
                scores = self.jnp.where(mask_arr, scores, neg_inf)
            else:
                scores = scores + mask_arr

        if sinks is not None:
            sinks_arr = self.jnp.asarray(sinks)
            scores = scores + sinks_arr

        # Softmax and apply to values
        attn_weights = self.jax.nn.softmax(scores, axis=-1)
        return self.jnp.einsum("...hqk,...khd->...qhd", attn_weights, v)

    # --- Stream Management ---

    def new_stream(self, device: str = "gpu") -> Any:
        """Create a new stream for parallel execution.

        Args:
            device: "gpu" or "cpu"

        Returns:
            None - JAX manages streams internally via XLA.
        """
        # JAX manages device placement internally via XLA.
        # No explicit stream creation needed.
        return None

    def synchronize(self) -> None:
        """Synchronize all computation (wait for all work to complete)."""
        # Block until all pending computations are complete
        self.jax.block_until_ready(self.jnp.array(0))

    # --- File I/O (Native Backend Serialization) ---

    def save_safetensors(
        self, path: str, weights: dict[str, Any], metadata: dict[str, str] | None = None
    ) -> None:
        """Save weights to safetensors using JAX/Flax native I/O.

        Args:
            path: File path to save to.
            weights: Dictionary of weight name -> array.
            metadata: Optional dictionary of string metadata to include.
        """
        from safetensors.flax import save_file

        # Ensure all arrays are JAX arrays
        jax_weights = {}
        for key, value in weights.items():
            if hasattr(value, "__module__") and "jax" in type(value).__module__:
                jax_weights[key] = value
            else:
                jax_weights[key] = self.array(value)
        if metadata:
            save_file(jax_weights, path, metadata=metadata)
        else:
            save_file(jax_weights, path)

    def load_safetensors(self, path: str) -> dict[str, Any]:
        """Load weights from safetensors using JAX/Flax native I/O.

        Args:
            path: File path to load from.

        Returns:
            Dictionary of weight name -> JAX array.
        """
        from safetensors.flax import load_file

        return load_file(path)

    # --- Model Operations ---

    def load_model(
        self, path: str, adapter_path: str | None = None
    ) -> tuple[Any, Any]:
        """Load a model and tokenizer using transformers.

        Args:
            path: Path to model directory.
            adapter_path: Optional path to LoRA adapter.

        Returns:
            Tuple of (model, tokenizer).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path)

        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)

        return model, tokenizer

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        """Generate text using transformers generate.

        Args:
            model: Model object from load_model.
            tokenizer: Tokenizer object from load_model.
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text string.
        """
        inputs = tokenizer(prompt, return_tensors="np")
        input_ids = self.jnp.array(inputs["input_ids"])
        # JAX models typically need custom generate - use transformers fallback
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_tokens,
            **kwargs,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_embed_tokens(self, model: Any) -> Any:
        """Get the embedding matrix from a model.

        Args:
            model: Model object from load_model.

        Returns:
            Embedding weight matrix [vocab_size, hidden_dim].
        """
        # For HuggingFace models
        if hasattr(model, "model"):
            base = model.model
        else:
            base = model

        if hasattr(base, "embed_tokens"):
            weight = base.embed_tokens.weight
        elif hasattr(base, "wte"):
            weight = base.wte.weight
        else:
            raise ValueError("Cannot find embedding weights in model")

        # Convert to JAX array
        return self.jnp.array(weight.detach().numpy())

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
        if hasattr(model, "config"):
            if hasattr(model.config, "num_hidden_layers"):
                return model.config.num_hidden_layers
            if hasattr(model.config, "n_layer"):
                return model.config.n_layer
        raise ValueError("Cannot determine number of layers")

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
        # Use transformers output_hidden_states
        activations: dict[int, list[Any]] = {}

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="np")
            outputs = model(
                input_ids=inputs["input_ids"],
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states
            n_layers = len(hidden_states) - 1  # Exclude embedding

            if layer_indices is None:
                layer_indices = list(range(n_layers))

            for layer_idx in layer_indices:
                if layer_idx not in activations:
                    activations[layer_idx] = []
                # hidden_states[0] is embedding, layers start at 1
                hs = hidden_states[layer_idx + 1]
                activations[layer_idx].append(self.jnp.array(hs))

        # Stack activations per layer
        result = {}
        for layer_idx, acts in activations.items():
            if acts:
                result[layer_idx] = self.jnp.concatenate(acts, axis=0)

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
        inputs = tokenizer(prompt, return_tensors="np")
        outputs = model(
            input_ids=inputs["input_ids"],
            output_hidden_states=True,
        )

        norms = []
        for hidden_state in outputs.hidden_states:
            hs = self.jnp.array(hidden_state)
            norm = float(self.jnp.sqrt(self.jnp.sum(hs * hs)))
            norms.append(norm)

        return norms

    # --- Neural Network Operations ---

    def silu(self, array: Any) -> Any:
        """SiLU (Swish) activation function: x * sigmoid(x)."""
        return self.jax.nn.silu(array)

    # --- Memory Management ---

    def get_peak_memory_gb(self) -> float:
        """Get peak GPU memory usage in gigabytes."""
        # JAX doesn't track peak memory in the same way
        # Return current allocation as approximation
        return self.get_active_memory_gb()

    def get_active_memory_gb(self) -> float:
        """Get active GPU memory usage in gigabytes."""
        try:
            devices = self.jax.devices()
            if devices:
                stats = devices[0].memory_stats()
                if stats:
                    return stats.get("bytes_in_use", 0) / (1024**3)
        except Exception:
            pass
        return 0.0

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
        input_ids = self.jnp.array([token_ids])

        outputs = model(input_ids=input_ids, output_hidden_states=True)
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            embedding = self.jnp.mean(outputs.hidden_states[0], axis=(0, 1))
            return embedding

        raise RuntimeError("Embedding extraction not supported for this JAX model")

    def collect_intermediate_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> dict[int, Any]:
        """Collect per-layer MLP intermediate activations."""
        # JAX intermediate extraction requires model surgery - return empty
        return {}

    def collect_attention_activations(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
        token_ids: list[int] | None = None,
    ) -> tuple[dict[int, Any], dict[int, Any], dict[int, Any]]:
        """Collect per-layer attention Q, K, V activations."""
        # JAX attention extraction requires model surgery - return empty
        return {}, {}, {}

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
        input_ids = self.jnp.array([token_ids])

        outputs = model(input_ids=input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        if logits.ndim == 3:
            last_logits = logits[0, -1, :]
        elif logits.ndim == 2:
            last_logits = logits[0, :]
        else:
            last_logits = logits

        return last_logits

    def collect_probe_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> Any:
        """Collect hidden + intermediate + gate + embedding activations in batch."""
        from modelcypher.ports.activation_provider import ProbeActivationBatch

        if not texts:
            return ProbeActivationBatch(hidden=[], intermediate=[], gate=[], embedding=[])

        # Sequential collection
        hidden: list[dict[int, Any]] = []
        intermediate: list[dict[int, Any]] = []
        gate: list[dict[int, Any]] = []
        embedding: list[Any] = []

        for text in texts:
            hidden.append(self._collect_hidden_single(model, tokenizer, text))
            intermediate.append({})  # Not supported
            gate.append({})  # Not supported
            embedding.append(self.collect_embedding_activations(model, tokenizer, text))

        return ProbeActivationBatch(
            hidden=hidden,
            intermediate=intermediate,
            gate=gate,
            embedding=embedding,
        )

    def _collect_hidden_single(
        self,
        model: Any,
        tokenizer: Any,
        text: str,
    ) -> dict[int, Any]:
        """Collect hidden activations for a single text."""
        token_ids = tokenizer.encode(text)
        input_ids = self.jnp.array([token_ids])

        outputs = model(input_ids=input_ids, output_hidden_states=True)

        activations = {}
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            for layer_idx, hidden in enumerate(outputs.hidden_states[1:]):  # Skip embedding
                activations[layer_idx] = self.jnp.mean(hidden, axis=(0, 1))

        return activations

    def collect_hidden_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Any]]:
        """Collect per-layer hidden activations for multiple texts."""
        return [self._collect_hidden_single(model, tokenizer, t) for t in texts]

    def collect_intermediate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Any]]:
        """Collect per-layer intermediate activations for multiple texts."""
        return [{} for _ in texts]  # Not supported in JAX

    def collect_gate_activations_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> list[dict[int, Any]]:
        """Collect per-layer gate activations for multiple texts."""
        return [{} for _ in texts]  # Not supported in JAX

    def collect_trajectory_batch(
        self,
        model: Any,
        tokenizer: Any,
        texts: list[str],
    ) -> Any:
        """Collect full trajectory activations for manifold mapping."""
        raise NotImplementedError("Trajectory collection not implemented for JAX backend")
