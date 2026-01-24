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

"""Backend protocol for GPU tensor operations.

Domain code depends on this protocol to keep computation on-device. When using
lazy backends (e.g., MLX), call `eval()` before extracting Python values.
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
    def triu_indices(self, n: int, k: int = 0) -> tuple[Array, Array]: ...
    def diag(self, array: Array, k: int = 0) -> Array: ...
    def full(
        self, shape: tuple[int, ...], fill_value: float, dtype: Any | None = None
    ) -> Array: ...
    def ones_like(self, array: Array, dtype: Any | None = None) -> Array: ...
    def zeros_like(self, array: Array, dtype: Any | None = None) -> Array: ...
    def linspace(self, start: float, stop: float, num: int, dtype: Any | None = None) -> Array: ...
    def meshgrid(self, *arrays: Array, indexing: str = "xy") -> list[Array]:
        """Create coordinate matrices from 1D coordinate arrays.

        Args:
            *arrays: 1D arrays representing the coordinates of a grid.
            indexing: Cartesian ('xy', default) or matrix ('ij') indexing.

        Returns:
            List of coordinate matrices for vectorized evaluations.
        """
        ...

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
    def tile(self, array: Array, reps: tuple[int, ...] | int) -> Array:
        """Construct an array by repeating the input the given number of times.

        Args:
            array: Input array.
            reps: Number of repetitions along each axis. If an int, tile
                  that many times along all axes.

        Returns:
            Tiled output array.
        """
        ...
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
    def all(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        """Test whether all elements evaluate to True along axis.

        Args:
            array: Input array.
            axis: Axis or axes along which to perform the reduction.
                  If None, reduces over all elements.
            keepdims: If True, retains reduced dimensions with size 1.

        Returns:
            Boolean array with True where all elements are True.
        """
        ...
    def any(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        """Test whether any element evaluates to True along axis.

        Args:
            array: Input array.
            axis: Axis or axes along which to perform the reduction.
                  If None, reduces over all elements.
            keepdims: If True, retains reduced dimensions with size 1.

        Returns:
            Boolean array with True where any element is True.
        """
        ...

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
    def mod(self, lhs: Array, rhs: Array | float | int) -> Array:
        """Element-wise modulus (remainder)."""
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
    def eigvalsh(self, array: Array) -> Array:
        """Compute eigenvalues of symmetric/Hermitian matrix (values only, ~50% faster than eigh)."""
        ...
    def solve(self, a: Array, b: Array) -> Array: ...
    def inv(self, array: Array) -> Array:
        """Compute the inverse of a square matrix."""
        ...

    def solve(self, a: Array, b: Array) -> Array:
        """Solve a linear matrix equation, or system of linear scalar equations.

        Computes x for a well-determined, full-rank system ax = b.
        """
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

    # --- Graph Algorithms ---
    def floyd_warshall(self, dist: Array) -> Array:
        """Compute all-pairs shortest paths via Floyd-Warshall.

        Args:
            dist: Square distance matrix [n, n]. Uses current values as initial edges.

        Returns:
            Array of shortest-path distances [n, n].
        """
        ...
    def single_source_shortest_paths(self, dist: Array, source_index: int) -> Array:
        """Compute shortest paths from a single source using a dense adjacency matrix.

        Args:
            dist: Square distance matrix [n, n] with non-negative edge weights.
            source_index: Index of the source node.

        Returns:
            Array of shortest-path distances [n] from source to all nodes.
        """
        ...

    # --- Transforms ---
    def vmap(
        self,
        fun: Callable,
        in_axes: int | tuple[int | None, ...] | None = 0,
        out_axes: int | tuple[int | None, ...] | None = 0,
    ) -> Callable:
        """Vectorize a function over batch axes."""
        ...

    def value_and_grad(self, fun: Callable, argnums: int | list[int] = 0) -> Callable:
        """Return a function that computes both value and gradient of fun.
        
        Args:
            fun: Function to differentiate.
            argnums: Argument index(es) to differentiate with respect to.
            
        Returns:
            Function taking same args as fun, returning (value, gradient).
        """
        ...

    # --- Indexing ---
    def take(self, array: Array, indices: Array, axis: int | None = None) -> Array: ...
    def take_along_axis(self, array: Array, indices: Array, axis: int) -> Array:
        """Take values from array along axis using indices array.

        Similar to numpy.take_along_axis. For each position in indices,
        gathers the value from array at that index along the specified axis.

        Args:
            array: Source array to gather from.
            indices: Index array (must broadcast with array except along axis).
            axis: Axis along which to gather.

        Returns:
            Array with same shape as indices, containing gathered values.
        """
        ...
    def put_along_axis(
        self, array: Array, indices: Array, values: Array, axis: int | None = None
    ) -> Array:
        """Put values into array along axis at the specified indices.

        Similar to numpy.put_along_axis. For each position in indices,
        writes the corresponding value into array along the specified axis.

        Args:
            array: Destination array to update.
            indices: Index array (broadcastable with array except along axis).
            values: Values array (broadcastable with indices).
            axis: Axis along which to put values, or None to flatten.

        Returns:
            Updated array with values written at indices.
        """
        ...

    def nonzero(self, array: Array) -> tuple[Array, ...]:
        """Find indices of non-zero elements.

        Returns a tuple of arrays, one for each dimension of the input,
        containing the indices of the non-zero elements in that dimension.

        For a 1D array, returns a tuple with a single array of indices.
        For a 2D array, returns (row_indices, col_indices).

        Args:
            array: Input array.

        Returns:
            Tuple of index arrays, one per dimension.
        """
        ...

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

    def randperm(self, n: int) -> Array:
        """Generate a random permutation of integers from 0 to n-1.

        Args:
            n: Number of elements in the permutation.

        Returns:
            1D array of shape (n,) containing a random permutation of [0, n).
        """
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

    def matrix_sqrt_newton_schulz(self, A: Array, num_iters: int = 15) -> Array:
        """Compute matrix square root iteratively via Newton-Schulz.
        
        Converges to A^{1/2} for positive semi-definite A.
        Runs entirely on GPU.
        """
        ...

    def async_eval(self, *arrays: Array) -> None:
        """Asynchronously evaluate arrays for pipeline parallelism.

        Allows CPU work to continue while GPU computes. Essential for
        overlapping data preparation with model inference.
        """
        ...

    # --- Fused Operations (Metal Kernels) ---
    def rms_norm(
        self, x: Array, weight: Array | None, eps: float = 1e-5, stream: Any | None = None
    ) -> Array:
        """Fused RMS normalization kernel.

        Combines sqrt, mean, and multiply into single kernel.
        Critical for transformer inference performance.
        Supports optional streams on backends that expose them.
        """
        ...

    def layer_norm(
        self,
        x: Array,
        weight: Array | None,
        bias: Array | None,
        eps: float = 1e-5,
        stream: Any | None = None,
    ) -> Array:
        """Fused Layer normalization kernel.

        Combines mean, variance, normalize, scale, and shift into single kernel.
        Supports optional streams on backends that expose them.
        """
        ...

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
        """Fused Rotary Position Embedding kernel.

        Applies rotary position embeddings in a single fused operation.
        Essential for efficient attention computation.
        If freqs is provided, base must be None.
        """
        ...

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
        """Fused Scaled Dot-Product Attention kernel (Flash-attention-style).

        Combines Q@K^T, scaling, masking, softmax, and @V into optimized kernel.
        Dramatically reduces memory bandwidth and improves throughput.
        mask may be "causal" or an additive/boolean mask.
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
