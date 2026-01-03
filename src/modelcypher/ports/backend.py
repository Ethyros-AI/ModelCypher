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
Backend Protocol for GPU tensor operations.

This is the abstract interface that keeps ALL computation on GPU. Domain code
depends on this protocol, never on concrete backends or NumPy.

Why No NumPy?
-------------
Every user has a GPU. NumPy forces CPU fallback and kills performance.
The Backend protocol ensures tensors stay on-device (Metal/CUDA/TPU)
throughout the entire computation pipeline.

    # WRONG - Forces CPU fallback, breaks performance
    import numpy as np
    mean = np.mean(vectors, axis=0)
    sorted_vals = np.sort(eigenvalues)[::-1]

    # CORRECT - Stays on GPU
    from modelcypher.core.domain._backend import get_default_backend
    backend = get_default_backend()
    mean = backend.mean(vectors, axis=0)
    sorted_idx = backend.argsort(eigenvalues)
    reversed_idx = backend.arange(n - 1, -1, -1)
    sorted_vals = backend.take(eigenvalues, reversed_idx, axis=0)

NumPy to Backend Migration Patterns
-----------------------------------
Common NumPy patterns and their Backend replacements:

+----------------------------------+------------------------------------------------------+
| NumPy Pattern                    | Backend Replacement                                  |
+==================================+======================================================+
| arr[::-1]                        | backend.take(arr, backend.arange(n-1, -1, -1), 0)    |
+----------------------------------+------------------------------------------------------+
| arr[mask]                        | backend.where(mask, arr, backend.zeros_like(arr))    |
+----------------------------------+------------------------------------------------------+
| np.sort(arr)                     | backend.sort(arr)                                    |
+----------------------------------+------------------------------------------------------+
| np.linalg.det(A)                 | backend.det(A)                                       |
+----------------------------------+------------------------------------------------------+
| for x in to_numpy(arr):          | Use backend.take(arr, idx, axis=0) for indexed access|
+----------------------------------+------------------------------------------------------+
| arr.tolist()                     | backend.tolist(arr)                                  |
+----------------------------------+------------------------------------------------------+
| result = np.array(python_list)   | result = backend.array(python_list)                  |
+----------------------------------+------------------------------------------------------+
| arr[:, -1] *= -1                 | scale = backend.array([1.0]*(d-1) + [-1.0])          |
|                                  | arr = arr * scale                                    |
+----------------------------------+------------------------------------------------------+

Numerical Stability
-------------------
All numerical thresholds MUST be derived from the dtype, never hardcoded.
Use the numerical_stability module:

    from modelcypher.core.domain.geometry.numerical_stability import (
        machine_epsilon,      # Smallest x where 1.0 + x != 1.0
        division_epsilon,     # Safe divisor threshold (~1e-7 for float32)
        regularization_epsilon,  # Matrix regularization (~1e-4 for float32)
    )

    # WRONG - Hardcoded magic number
    if denominator < 1e-8:
        return 0.0

    # CORRECT - Derived from dtype
    eps = division_epsilon(backend, arr)
    if denominator < eps:
        return 0.0

FloatInfo provides raw precision info:

    info = backend.finfo(arr.dtype)
    info.eps   # Machine epsilon
    info.tiny  # Smallest positive usable number
    info.max   # Largest finite number
    info.min   # Most negative finite number

Lazy Evaluation (MLX)
---------------------
MLX uses lazy evaluation - computations are recorded but not executed until
explicitly evaluated. This enables kernel fusion but requires care:

    # WRONG - Lazy array passed to Python, causes crash or wrong value
    mean = backend.mean(arr)
    print(float(mean))  # May print garbage or crash

    # CORRECT - Evaluate before extracting Python values
    mean = backend.mean(arr)
    backend.eval(mean)
    print(backend.to_scalar(mean))

When to call eval():
    - Before to_scalar(): Always eval() first
    - Before tolist(): Always eval() first
    - Before conditional logic: if float(to_scalar(x)) > threshold
    - Between pipeline stages: eval() to materialize intermediate results
    - NOT needed: Between pure backend operations (let MLX fuse them)

The pattern is: compute -> eval -> extract

    result = backend.sum(backend.exp(arr))  # Lazy
    backend.eval(result)                     # Force computation
    value = backend.to_scalar(result)        # Safe to extract

Adding New Operations
---------------------
When the Backend doesn't have an operation you need, ADD IT. Don't work around
missing ops with NumPy.

Steps to add a new operation:

1. Add signature to Backend protocol (this file):

    def my_operation(self, array: Array, param: float) -> Array:
        '''Docstring explaining the operation.'''
        ...

2. Implement in MLXBackend (src/modelcypher/backends/mlx_backend.py):

    def my_operation(self, array: Array, param: float) -> Array:
        import mlx.core as mx
        return mx.my_mlx_equivalent(array, param)

3. Implement in JAXBackend (src/modelcypher/backends/jax_backend.py):

    def my_operation(self, array: Array, param: float) -> Array:
        import jax.numpy as jnp
        return jnp.my_jax_equivalent(array, param)

4. Add tests (tests/test_backends.py):

    def test_my_operation(backend):
        arr = backend.array([1.0, 2.0, 3.0])
        result = backend.my_operation(arr, 0.5)
        expected = ...
        assert_allclose(backend.to_numpy(result), expected)

Available Implementations
-------------------------
- MLXBackend: Primary backend for macOS (Metal GPU acceleration)
- JAXBackend: Secondary backend for Linux/TPU
- NumPyBackend: Testing only - never use in production

Usage:

    from modelcypher.core.domain._backend import get_default_backend

    backend = get_default_backend()  # Auto-selects MLX on Mac, JAX on Linux
    arr = backend.array([[1, 2], [3, 4]])
    result = backend.matmul(arr, backend.transpose(arr))

Architecture Note
-----------------
This protocol is a PORT in hexagonal architecture. Domain code imports Backend,
never concrete implementations. The adapters (MLXBackend, JAXBackend) are
injected at runtime based on platform detection.

    core/domain/ --> ports/Backend (this file) <-- backends/mlx_backend.py
                                               <-- backends/jax_backend.py

Dependencies point inward. Domain never imports backends directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable

# TypeVar for array types - provides documentation and enables future typing improvements
# while remaining compatible with MLX arrays, NumPy ndarrays, and other backends.
# Using TypeVar instead of Any signals that these are homogeneous array operations.
Array = TypeVar("Array")


@dataclass(frozen=True)
class FloatInfo:
    """Floating-point precision information derived from dtype.

    All numerical stability constants should be derived from these values,
    not hardcoded magic numbers.
    """

    eps: float  # Machine epsilon: smallest x where 1.0 + x != 1.0
    tiny: float  # Smallest positive usable number
    max: float  # Largest representable finite number
    min: float  # Most negative representable finite number


@runtime_checkable
class Backend(Protocol):
    """
    Abstract backend protocol for tensor operations.

    Implementations: MLXBackend (macOS), JAXBackend (Linux/TPU), CUDABackend (NVIDIA).
    Domain classes should depend on this protocol, not concrete backends.
    """

    # --- Array Creation ---
    def array(self, data: Any, dtype: Any | None = None) -> Array: ...
    def zeros(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array: ...
    def ones(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array: ...
    def eye(self, n: int, m: int | None = None, dtype: Any | None = None) -> Array: ...
    def arange(
        self,
        start: int | float,
        stop: int | float | None = None,
        step: int | float = 1,
        dtype: Any | None = None,
    ) -> Array: ...
    def diag(self, array: Array, k: int = 0) -> Array: ...
    def full(
        self, shape: tuple[int, ...], fill_value: float, dtype: Any | None = None
    ) -> Array: ...
    def ones_like(self, array: Array, dtype: Any | None = None) -> Array: ...
    def zeros_like(self, array: Array, dtype: Any | None = None) -> Array: ...
    def linspace(self, start: float, stop: float, num: int, dtype: Any | None = None) -> Array: ...

    # --- Shape Manipulation ---
    def shape(self, array: Array) -> tuple[int, ...]:
        """Return the shape of an array."""
        ...
    def reshape(self, array: Array, shape: tuple[int, ...]) -> Array: ...
    def squeeze(self, array: Array, axis: int | None = None) -> Array: ...
    def transpose(self, array: Array, axes: tuple[int, ...] | None = None) -> Array: ...
    def stack(self, arrays: list[Array], axis: int = 0) -> Array: ...
    def concatenate(self, arrays: list[Array], axis: int = 0) -> Array: ...
    def broadcast_to(self, array: Array, shape: tuple[int, ...]) -> Array: ...
    def expand_dims(self, array: Array, axis: int | tuple[int, ...]) -> Array:
        """Insert a new axis (dimension of size 1) at the specified position.

        Args:
            array: Input array.
            axis: Position(s) where new axis should be inserted.

        Returns:
            Array with expanded dimensions.
        """
        ...

    # --- Reductions ---
    def sum(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array: ...
    def mean(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array: ...
    def max(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array: ...
    def min(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array: ...
    def argmax(self, array: Array, axis: int | None = None) -> Array: ...
    def argmin(self, array: Array, axis: int | None = None) -> Array: ...
    def var(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array: ...
    def std(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array: ...

    # --- Element-wise Operations ---
    def sqrt(self, array: Array) -> Array: ...
    def exp(self, array: Array) -> Array: ...
    def log(self, array: Array) -> Array: ...
    def abs(self, array: Array) -> Array: ...
    def sign(self, array: Array) -> Array: ...
    def isnan(self, array: Array) -> Array:
        """Element-wise test for NaN (Not a Number).

        Returns a boolean array with True where elements are NaN.
        """
        ...
    def isinf(self, array: Array) -> Array:
        """Element-wise test for infinity (positive or negative).

        Returns a boolean array with True where elements are +/- inf.
        """
        ...
    def isfinite(self, array: Array) -> Array:
        """Element-wise test for finite values.

        Returns a boolean array with True where elements are neither NaN nor inf.
        """
        ...
    def sin(self, array: Array) -> Array:
        """Element-wise sine (radians)."""
        ...
    def cos(self, array: Array) -> Array:
        """Element-wise cosine (radians)."""
        ...
    def arccos(self, array: Array) -> Array:
        """Element-wise inverse cosine, returns radians in [0, π]."""
        ...
    def arctan(self, array: Array) -> Array:
        """Element-wise inverse tangent, returns radians in [-π/2, π/2]."""
        ...
    def lgamma(self, array: Array) -> Array:
        """Element-wise log-gamma function (ln(gamma(x)))."""
        ...
    def maximum(self, lhs: Array, rhs: Array) -> Array: ...
    def minimum(self, lhs: Array, rhs: Array) -> Array: ...
    def clip(
        self, array: Array, min_val: float | Array | None, max_val: float | Array | None
    ) -> Array: ...
    def where(self, condition: Array, x: Array, y: Array) -> Array: ...
    def softmax(self, array: Array, axis: int = -1) -> Array: ...
    def cumsum(self, array: Array, axis: int | None = None) -> Array: ...
    def floor(self, array: Array) -> Array:
        """Element-wise floor (round toward negative infinity)."""
        ...
    def ceil(self, array: Array) -> Array:
        """Element-wise ceiling (round toward positive infinity)."""
        ...
    def log2(self, array: Array) -> Array:
        """Element-wise base-2 logarithm."""
        ...

    # --- Linear Algebra ---
    def matmul(self, lhs: Array, rhs: Array) -> Array: ...
    def dot(self, a: Array, b: Array) -> Array: ...
    def svd(self, array: Array, compute_uv: bool = True) -> tuple[Array, Array, Array] | Array: ...
    def norm(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array: ...
    def det(self, array: Array) -> Array: ...
    def eigh(self, array: Array) -> tuple[Array, Array]: ...
    def solve(self, a: Array, b: Array) -> Array: ...
    def inv(self, array: Array) -> Array:
        """Compute the inverse of a square matrix."""
        ...
    def pinv(self, array: Array) -> Array:
        """Compute the Moore-Penrose pseudoinverse of a matrix."""
        ...
    def cholesky(self, array: Array) -> Array:
        """Compute the Cholesky decomposition of a positive-definite matrix."""
        ...
    def trace(self, array: Array) -> Array:
        """Compute the trace (sum of diagonal elements) of a matrix."""
        ...
    def qr(self, array: Array) -> tuple[Array, Array]: ...

    # --- Indexing ---
    def take(self, array: Array, indices: Array, axis: int | None = None) -> Array: ...

    # --- Sorting ---
    def sort(self, array: Array, axis: int = -1) -> Array: ...
    def argsort(self, array: Array, axis: int = -1) -> Array: ...
    def argpartition(self, array: Array, kth: int, axis: int = -1) -> Array: ...
    def partition(self, array: Array, kth: int, axis: int = -1) -> Array:
        """Partition array so kth element is in sorted position (O(n) complexity).

        After partitioning, array[kth] equals what it would be in a sorted array.
        Elements before kth are <= array[kth], elements after are >= array[kth].
        Use for efficient percentile computation without full O(n log n) sort.
        """
        ...

    # --- Random ---
    def random_normal(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array: ...
    def random_uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        shape: tuple[int, ...] | None = None,
        dtype: Any | None = None,
    ) -> Array: ...
    def random_randint(
        self, low: int, high: int, shape: tuple[int, ...] | None = None
    ) -> Array: ...
    def random_seed(self, seed: int) -> None: ...
    def random_categorical(self, logits: Array, num_samples: int = 1) -> Array:
        """Sample from categorical distribution defined by logits."""
        ...

    # --- Type Conversion ---
    def astype(self, array: Array, dtype: Any) -> Array: ...
    def to_numpy(self, array: Array) -> Any: ...
    def to_scalar(self, array: Array) -> float | int:
        """Extract a scalar from a 0-d or single-element array.

        Faster than to_numpy().item() - skips numpy conversion entirely.
        Array must be evaluated first (call eval() before to_scalar()).

        Args:
            array: A scalar (0-d) or single-element array.

        Returns:
            Python float or int.

        Raises:
            ValueError: If array has more than one element.
        """
        ...
    def tolist(self, array: Array) -> list | float | int:
        """Convert array to nested Python lists.

        This is MUCH faster than element-by-element to_scalar() extraction.
        Uses native backend tolist() which avoids Python loops entirely.

        Args:
            array: Array of any shape.

        Returns:
            Nested Python list matching array shape, or scalar for 0-d arrays.
        """
        ...
    def finfo(self, dtype: Any | None = None) -> FloatInfo:
        """Return floating-point precision info for the given dtype.

        If dtype is None, uses the backend's default float type (typically float32).
        All numerical stability constants should derive from this, not hardcoded values.
        """
        ...

    # --- Quantization ---
    def quantize(
        self,
        weight: Array,
        group_size: int,
        bits: int,
        mode: str,
    ) -> tuple[Array, Array, Array | None]: ...
    def dequantize(
        self,
        weight: Array,
        scales: Array,
        biases: Array | None,
        group_size: int,
        bits: int,
        mode: str,
    ) -> Array: ...

    # --- Attention Masks ---
    def create_causal_mask(self, seq_len: int, dtype: Any | None = None) -> Array:
        """Create an additive causal attention mask for autoregressive models."""
        ...

    # --- Compute Control ---
    def eval(self, *arrays: Array) -> None: ...

    def clear_cache(self) -> None:
        """Clear GPU memory cache to release lazy computations.

        Essential between pipeline stages to prevent OOM from accumulated
        lazy computations. No-op on backends without explicit cache management.
        """
        ...

    # --- Performance APIs (SOTA MLX Features) ---
    def compile(
        self,
        fun: Callable,
        inputs: list | None = None,
        outputs: list | None = None,
        shapeless: bool = False,
    ) -> Callable:
        """JIT-compile a function for kernel fusion (5x speedup on MLX).

        Fuses element-wise operations into single Metal kernels.
        Falls back to identity on backends that don't support compilation.
        """
        ...

    def vmap(
        self,
        fun: Callable,
        in_axes: int | tuple | None = 0,
        out_axes: int | tuple | None = 0,
    ) -> Callable:
        """Auto-vectorize a function over batch dimension (200x speedup vs loops).

        Transforms a function that operates on single examples to operate on batches.
        Falls back to manual loop on backends that don't support vectorization.
        """
        ...

    def async_eval(self, *arrays: Array) -> None:
        """Asynchronously evaluate arrays for pipeline parallelism.

        Allows CPU work to continue while GPU computes. Essential for
        overlapping data preparation with model inference.
        """
        ...

    # --- Fused Operations (Metal Kernels) ---
    def rms_norm(self, x: Array, weight: Array, eps: float = 1e-5) -> Array:
        """Fused RMS normalization kernel.

        Combines sqrt, mean, and multiply into single kernel.
        Critical for transformer inference performance.
        """
        ...

    def layer_norm(
        self, x: Array, weight: Array | None, bias: Array | None, eps: float = 1e-5
    ) -> Array:
        """Fused Layer normalization kernel.

        Combines mean, variance, normalize, scale, and shift into single kernel.
        """
        ...

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

        Applies rotary position embeddings in a single fused operation.
        Essential for efficient attention computation.
        """
        ...

    def scaled_dot_product_attention(
        self,
        q: Array,
        k: Array,
        v: Array,
        scale: float,
        mask: Array | None = None,
    ) -> Array:
        """Fused Scaled Dot-Product Attention kernel (Flash-attention-style).

        Combines Q@K^T, scaling, masking, softmax, and @V into optimized kernel.
        Dramatically reduces memory bandwidth and improves throughput.
        """
        ...

    # --- Stream Management ---
    def new_stream(self, device: str = "gpu") -> Any:
        """Create a new stream for parallel execution.

        Enables CPU/GPU parallelism for data loading + compute overlap.
        """
        ...

    def synchronize(self) -> None:
        """Synchronize all streams.

        Waits for all pending GPU operations to complete.
        """
        ...

    # --- File I/O (Native Backend Serialization) ---
    def save_safetensors(
        self, path: str, weights: dict[str, Array], metadata: dict[str, str] | None = None
    ) -> None:
        """Save weights dictionary to safetensors format using native backend I/O.

        This respects hexagonal architecture - the backend decides the optimal
        serialization format for its array type.

        Args:
            path: File path to save to.
            weights: Dictionary of weight name -> array.
            metadata: Optional dictionary of string metadata to include.
        """
        ...

    def load_safetensors(self, path: str) -> dict[str, Array]:
        """Load weights from safetensors format using native backend I/O.

        Args:
            path: File path to load from.

        Returns:
            Dictionary of weight name -> backend array.
        """
        ...
