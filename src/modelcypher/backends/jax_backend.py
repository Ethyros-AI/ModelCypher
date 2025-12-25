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

JAX is ideal for high-dimensional geometry work due to:
- XLA compilation for efficient tensor operations
- Automatic differentiation for Jacobian/Hessian computation
- Functional purity enabling reproducible research
"""

from __future__ import annotations

from typing import Any

import numpy as _np_interop  # Interop boundary: Backend protocol requires to_numpy() and dtype mapping

from modelcypher.ports.backend import Array, Backend


class JAXBackend(Backend):
    """JAX implementation of the Backend protocol.

    Uses jax.numpy for array operations and jax.scipy.linalg for
    linear algebra. Random operations use explicit PRNG keys for
    reproducibility.

    Example:
        backend = JAXBackend()
        a = backend.array([[1, 2], [3, 4]])
        u, s, vt = backend.svd(a)
    """

    def __init__(self) -> None:
        import jax
        import jax.numpy as jnp

        self.jax = jax
        self.jnp = jnp
        self._rng_key = jax.random.PRNGKey(0)

    def _next_key(self) -> Any:
        """Get next PRNG key and update internal state."""
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

    def diag(self, array: Array, k: int = 0) -> Array:
        return self.jnp.diag(array, k=k)

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: Any | None = None) -> Array:
        return self.jnp.full(shape, fill_value, dtype=self._map_dtype(dtype))

    def ones_like(self, array: Array, dtype: Any | None = None) -> Array:
        return self.jnp.ones_like(array, dtype=self._map_dtype(dtype))

    def zeros_like(self, array: Array, dtype: Any | None = None) -> Array:
        return self.jnp.zeros_like(array, dtype=self._map_dtype(dtype))

    def linspace(self, start: float, stop: float, num: int, dtype: Any | None = None) -> Array:
        return self.jnp.linspace(start, stop, num, dtype=self._map_dtype(dtype))

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

    def maximum(self, lhs: Array, rhs: Array) -> Array:
        return self.jnp.maximum(lhs, rhs)

    def minimum(self, lhs: Array, rhs: Array) -> Array:
        return self.jnp.minimum(lhs, rhs)

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

    def eigh(self, array: Array) -> tuple[Array, Array]:
        eigenvalues, eigenvectors = self.jnp.linalg.eigh(array)
        return eigenvalues, eigenvectors

    def solve(self, a: Array, b: Array) -> Array:
        return self.jnp.linalg.solve(a, b)

    def inv(self, array: Array) -> Array:
        return self.jnp.linalg.inv(array)

    def cholesky(self, array: Array) -> Array:
        return self.jnp.linalg.cholesky(array)

    def trace(self, array: Array) -> Array:
        return self.jnp.trace(array)

    def qr(self, array: Array) -> tuple[Array, Array]:
        q, r = self.jnp.linalg.qr(array)
        return q, r

    # --- Indexing ---
    def take(self, array: Array, indices: Array, axis: int | None = None) -> Array:
        return self.jnp.take(array, indices, axis=axis)

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
        """O(n) partition via jnp.partition (uses jax.lax.top_k internally)."""
        return self.jnp.partition(array, kth=kth, axis=axis)

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

    # --- Type Conversion ---
    def astype(self, array: Array, dtype: Any) -> Array:
        return array.astype(self._map_dtype(dtype))

    def to_numpy(self, array: Array) -> Any:
        return _np_interop.asarray(array)

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

    def _map_dtype(self, dtype: Any | None) -> Any | None:
        if dtype is None:
            return None
        # Handle string dtype names
        if isinstance(dtype, str):
            dtype_map = {
                "float32": self.jnp.float32,
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
        # Handle numpy dtype constants
        if dtype is _np_interop.float32:
            return self.jnp.float32
        if dtype is _np_interop.float16:
            return self.jnp.float16
        if dtype is _np_interop.int32:
            return self.jnp.int32
        if dtype is _np_interop.int64:
            return self.jnp.int64
        return dtype
