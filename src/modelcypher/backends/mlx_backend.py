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

import numpy as _np_interop  # Interop boundary: Backend protocol requires to_numpy() and dtype mapping

from modelcypher.backends.safe_gpu import SafeGPU
from modelcypher.ports.backend import Array, Backend

if TYPE_CHECKING:
    pass


class MLXBackend(Backend):
    """MLX backend with lazy evaluation and SOTA performance optimizations.

    Key Performance APIs:
    - compile(): JIT-compile functions for kernel fusion (5x speedup)
    - fast: Access to mx.fast.* fused Metal kernels
    - vmap(): Auto-vectorize functions over batch dimension
    - async_eval(): Non-blocking evaluation for pipeline parallelism

    IMPORTANT: Operations do NOT call eval() - this enables MLX's lazy evaluation
    and kernel fusion. Callers must explicitly call eval() when results are needed.
    """

    def __init__(self) -> None:
        import mlx.core as mx

        self.mx = mx
        self.fast = mx.fast  # Expose mx.fast for fused kernels
        self.safe = SafeGPU(mx)
        self._compiled_cache: dict[str, Callable] = {}

    # --- Array Creation (lazy - no eval) ---
    def array(self, data: Any, dtype: Any | None = None) -> Array:
        return self.mx.array(data, dtype=self._map_dtype(dtype))

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
        return array.astype(self._map_dtype(dtype))

    # --- Linear Algebra (requires eval for CPU stream ops) ---
    def svd(self, array: Array, compute_uv: bool = True) -> tuple[Array, Array, Array] | Array:
        # MLX SVD requires CPU stream - must eval before returning
        result = self.mx.linalg.svd(array, compute_uv=compute_uv, stream=self.mx.cpu)
        if compute_uv:
            u, s, vt = result
            self.safe.eval(u, s, vt)
            return u, s, vt
        self.safe.eval(result)
        return result

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

    def create_causal_mask(self, seq_len: int, dtype: Any | None = None) -> Array:
        """Create additive causal attention mask for autoregressive models."""
        import mlx.nn as nn

        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        if dtype is not None:
            mask = mask.astype(self._map_dtype(dtype))
        return mask

    def to_numpy(self, array: Array) -> Any:
        """Convert to numpy - requires eval first."""
        self.safe.eval(array)
        # Handle bfloat16 which numpy doesn't support natively
        if array.dtype == self.mx.bfloat16:
            array = array.astype(self.mx.float32)
            self.safe.eval(array)
        return _np_interop.array(array)

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

    def diag(self, array: Array, k: int = 0) -> Array:
        return self.mx.diag(array, k=k)

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

    # --- Shape Manipulation (lazy - no eval) ---
    def stack(self, arrays: list[Array], axis: int = 0) -> Array:
        return self.mx.stack(arrays, axis=axis)

    def concatenate(self, arrays: list[Array], axis: int = 0) -> Array:
        return self.mx.concatenate(arrays, axis=axis)

    def broadcast_to(self, array: Array, shape: tuple[int, ...]) -> Array:
        return self.mx.broadcast_to(array, shape)

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

    # --- Linear Algebra (lazy except CPU stream ops) ---
    def dot(self, a: Array, b: Array) -> Array:
        # MLX uses matmul for general case; for 1D vectors use sum of element-wise product
        if a.ndim == 1 and b.ndim == 1:
            return self.mx.sum(a * b)
        return self.mx.matmul(a, b)

    def norm(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
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

        n = int(p.shape[0])
        p_list = [int(x) for x in self.to_numpy(p).tolist()]
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

    def eigh(self, array: Array) -> tuple[Array, Array]:
        # MLX eigh requires CPU stream - must eval
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
        """O(n) partitioning for efficient percentile computation on GPU."""
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
        # Handle numpy dtype constants
        if dtype is _np_interop.float32:
            return self.mx.float32
        if dtype is _np_interop.float16:
            return self.mx.float16
        if dtype is _np_interop.int32:
            return self.mx.int32
        if dtype is _np_interop.int64:
            return self.mx.int64
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
        """JIT-compile a function for kernel fusion (5x speedup).

        Compiled functions fuse element-wise operations into single Metal kernels.
        The first call compiles; subsequent calls use cached compiled code.

        Args:
            fun: Function to compile
            inputs: Optional list of arrays to capture as implicit inputs
            outputs: Optional list of arrays to capture as implicit outputs
            shapeless: If True, don't recompile on shape changes (use carefully)

        Returns:
            Compiled function with identical behavior but fused kernels

        Example:
            # Before: 6 separate Metal kernels
            def gelu(x):
                return x * (1 + mx.erf(x / math.sqrt(2))) / 2

            # After: 1 fused Metal kernel (5x faster)
            fast_gelu = backend.compile(gelu)
        """
        return self.mx.compile(fun, inputs=inputs, outputs=outputs, shapeless=shapeless)

    def vmap(
        self,
        fun: Callable,
        in_axes: int | tuple | None = 0,
        out_axes: int | tuple | None = 0,
    ) -> Callable:
        """Auto-vectorize a function over batch dimension (200x speedup vs loops).

        Transforms a function that operates on single examples into one that
        efficiently processes batches using vectorized Metal operations.

        Args:
            fun: Function to vectorize
            in_axes: Which axis of each input to vectorize over (None = don't vectorize)
            out_axes: Where to place the batch axis in outputs

        Returns:
            Vectorized function

        Example:
            # Before: Slow loop over 4096 vectors
            def naive(xs, ys):
                return [xs[i] + ys[:, i] for i in range(xs.shape[0])]

            # After: Single vectorized operation (200x faster)
            fast = backend.vmap(lambda x, y: x + y, in_axes=(0, 1))
        """
        return self.mx.vmap(fun, in_axes=in_axes, out_axes=out_axes)

    def async_eval(self, *arrays: Array) -> None:
        """Asynchronously evaluate arrays for pipeline parallelism.

        Unlike eval(), this returns immediately while GPU work continues.
        Use for overlapping CPU preparation with GPU computation.

        Example:
            # Pipeline: prepare next batch while GPU processes current
            for batch in batches:
                backend.async_eval(result)  # Don't block
                next_input = prepare_next(batch)  # CPU work overlaps GPU
                result = model(next_input)
            backend.eval(result)  # Final sync
        """
        self.mx.async_eval(*arrays)

    # --- Fused Metal Kernels (mx.fast.*) ---

    def rms_norm(self, x: Array, weight: Array, eps: float = 1e-5) -> Array:
        """Fused RMS normalization kernel.

        Single Metal kernel instead of 5+ separate operations.
        Standard in Llama, Qwen, Mistral architectures.
        """
        return self.mx.fast.rms_norm(x, weight, eps)

    def layer_norm(
        self, x: Array, weight: Array | None, bias: Array | None, eps: float = 1e-5
    ) -> Array:
        """Fused Layer normalization kernel.

        Single Metal kernel for LayerNorm (GPT, BERT architectures).
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
        """Fused Rotary Position Embedding kernel.

        Single Metal kernel for RoPE (used in most modern LLMs).
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
        """Fused Scaled Dot-Product Attention kernel.

        Flash-attention-style fused kernel: O = softmax(Q @ K.T / scale) @ V
        Avoids materializing the full attention matrix.
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
