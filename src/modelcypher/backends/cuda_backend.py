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

from __future__ import annotations

from typing import Any, Callable

from modelcypher.backends.conversion_utils import (
    raise_numpy_disabled,
    to_list_with_eval,
    to_scalar_with_eval,
)
from modelcypher.ports.backend import Array, Backend, FloatInfo


class CUDABackend(Backend):
    def __init__(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for the CUDA backend") from exc
        self.torch = torch
        self._compiled_cache: dict[str, Callable] = {}

    def _tensor(self, data: Any, dtype: Any | None = None) -> Array:
        mapped_dtype = self._map_dtype(dtype)
        if isinstance(data, self.torch.Tensor):
            tensor = data
            if tensor.device.type != "cuda":
                tensor = tensor.to(device="cuda")
            if mapped_dtype is not None and tensor.dtype != mapped_dtype:
                tensor = tensor.to(dtype=mapped_dtype)
            return tensor
        resolved = mapped_dtype or self.torch.float32
        return self.torch.tensor(data, dtype=resolved, device="cuda")

    def array(self, data: Any, dtype: Any | None = None) -> Array:
        return self._tensor(data, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        resolved = self._map_dtype(dtype) or self.torch.float32
        return self.torch.zeros(shape, dtype=resolved, device="cuda")

    def ones(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        resolved = self._map_dtype(dtype) or self.torch.float32
        return self.torch.ones(shape, dtype=resolved, device="cuda")

    def shape(self, array: Array) -> tuple[int, ...]:
        return tuple(array.shape)

    def reshape(self, array: Array, shape: tuple[int, ...]) -> Array:
        return array.reshape(shape)

    def squeeze(self, array: Array, axis: int | None = None) -> Array:
        return array.squeeze(dim=axis) if axis is not None else array.squeeze()

    def transpose(self, array: Array, axes: tuple[int, ...] | None = None) -> Array:
        if axes is not None:
            return array.permute(axes)
        return array.t() if array.ndim == 2 else array.transpose(-2, -1)

    def matmul(self, lhs: Array, rhs: Array) -> Array:
        return lhs @ rhs

    def sum(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return array.sum(dim=axis, keepdim=keepdims)

    def max(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return array.max(dim=axis, keepdim=keepdims).values if axis is not None else array.max()

    def sqrt(self, array: Array) -> Array:
        return array.sqrt()

    def exp(self, array: Array) -> Array:
        return array.exp()

    def log(self, array: Array) -> Array:
        return array.log()

    def maximum(self, lhs: Array, rhs: Array) -> Array:
        return self.torch.maximum(lhs, rhs)

    def minimum(self, lhs: Array, rhs: Array) -> Array:
        return self.torch.minimum(lhs, rhs)

    def add(self, lhs: Array | float, rhs: Array | float) -> Array:
        """Element-wise addition."""
        return self.torch.add(lhs, rhs)

    def subtract(self, lhs: Array | float, rhs: Array | float) -> Array:
        """Element-wise subtraction."""
        return self.torch.subtract(lhs, rhs)

    def multiply(self, lhs: Array | float, rhs: Array | float) -> Array:
        """Element-wise multiplication."""
        return self.torch.multiply(lhs, rhs)

    def divide(self, lhs: Array | float, rhs: Array | float) -> Array:
        """Element-wise division."""
        return self.torch.divide(lhs, rhs)

    def abs(self, array: Array) -> Array:
        return array.abs()

    def astype(self, array: Array, dtype: Any) -> Array:
        return array.to(self._map_dtype(dtype) or dtype)

    def svd(self, array: Array, compute_uv: bool = True) -> tuple[Array, Array, Array] | Array:
        if compute_uv:
            u, s, vt = self.torch.linalg.svd(array, full_matrices=False)
            return u, s, vt
        return self.torch.linalg.svdvals(array)

    def quantize(
        self,
        weight: Array,
        group_size: int,
        bits: int,
        mode: str,
    ) -> tuple[Array, Array, Array | None]:
        """Quantize weights to lower precision.

        Parameters
        ----------
        weight : Array
            Weight tensor to quantize.
        group_size : int
            Number of elements per quantization group.
        bits : int
            Bit width for quantization (e.g., 4, 8).
        mode : str
            Quantization mode (e.g., 'affine', 'symmetric').

        Returns
        -------
        tuple[Array, Array, Array | None]
            Quantized weights, scales, and optional biases.
        """
        shape = weight.shape
        if len(shape) < 2:
            weight = weight.reshape(-1, 1)

        num_groups = weight.shape[0] // group_size
        weight_grouped = weight.reshape(num_groups, group_size, -1)

        # Compute scales per group
        max_vals = self.torch.amax(self.torch.abs(weight_grouped), dim=1, keepdim=True)
        scales = max_vals / (2 ** (bits - 1) - 1)
        scales = self.torch.where(scales == 0, self.torch.ones_like(scales), scales)

        # Quantize
        weight_q = self.torch.round(weight_grouped / scales)
        weight_q = self.torch.clamp(weight_q, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        weight_q = weight_q.to(dtype=self.torch.int8)

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
        """Dequantize weights back to full precision.

        Parameters
        ----------
        weight : Array
            Quantized weight tensor.
        scales : Array
            Scale factors per group.
        biases : Array | None
            Optional bias terms per group.
        group_size : int
            Number of elements per quantization group.
        bits : int
            Bit width used in quantization.
        mode : str
            Quantization mode (e.g., 'affine', 'symmetric').

        Returns
        -------
        Array
            Dequantized weight tensor.
        """
        shape = weight.shape
        weight = weight.to(dtype=self.torch.float32)

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

    def eval(self, *arrays: Array) -> None:
        self.torch.cuda.synchronize()

    def to_numpy(self, array: Array) -> Any:
        """DISABLED: CPU arrays are not permitted in ModelCypher.

        Use backend.tolist() or backend.to_scalar() for extracting values.
        Use backend.save_safetensors() for serialization.
        """
        raise_numpy_disabled()

    def to_scalar(self, array: Array) -> float | int:
        """Extract a scalar from a 0-d or single-element tensor.

        Faster than to_numpy().item() - uses PyTorch's native .item() directly,
        skipping CPU transfer and numpy conversion.

        Args:
            array: A scalar (0-d) or single-element tensor.

        Returns:
            Python float or int.

        Raises:
            ValueError: If tensor has more than one element.
        """
        return to_scalar_with_eval(array, self.eval)

    def tolist(self, array: Array) -> list | float | int:
        """Convert tensor to nested Python lists.

        Uses PyTorch's native tolist() - MUCH faster than element-by-element to_scalar().
        """
        return to_list_with_eval(array, self.eval)

    def finfo(self, dtype: Any | None = None) -> FloatInfo:
        """Return floating-point precision info for the given dtype.

        Derives numerical stability constants from the actual dtype precision.
        """
        resolved = dtype or self.torch.float32
        info = self.torch.finfo(resolved)
        return FloatInfo(
            eps=float(info.eps),
            tiny=float(info.tiny),
            max=float(info.max),
            min=float(info.min),
        )

    # --- Array Creation (new) ---
    def eye(self, n: int, m: int | None = None, dtype: Any | None = None) -> Array:
        resolved = self._map_dtype(dtype) or self.torch.float32
        return self.torch.eye(n, m or n, dtype=resolved, device="cuda")

    def arange(
        self,
        start: int | float,
        stop: int | float | None = None,
        step: int | float = 1,
        dtype: Any | None = None,
    ) -> Array:
        resolved = self._map_dtype(dtype)
        if stop is None:
            return self.torch.arange(start, dtype=resolved, device="cuda")
        return self.torch.arange(start, stop, step, dtype=resolved, device="cuda")

    def triu_indices(self, n: int, k: int = 0) -> tuple[Array, Array]:
        indices = self.torch.triu_indices(n, n, offset=k, device="cuda")
        return indices[0], indices[1]

    def diag(self, array: Array, k: int = 0) -> Array:
        return self.torch.diag(array, diagonal=k)

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: Any | None = None) -> Array:
        resolved = self._map_dtype(dtype) or self.torch.float32
        return self.torch.full(shape, fill_value, dtype=resolved, device="cuda")

    def dtype(self, array: Array) -> Any:
        """Return the dtype of an array."""
        return array.dtype

    def ones_like(self, array: Array, dtype: Any | None = None) -> Array:
        resolved = self._map_dtype(dtype)
        return self.torch.ones_like(array, dtype=resolved)

    def zeros_like(self, array: Array, dtype: Any | None = None) -> Array:
        resolved = self._map_dtype(dtype)
        return self.torch.zeros_like(array, dtype=resolved)

    def linspace(self, start: float, stop: float, num: int, dtype: Any | None = None) -> Array:
        resolved = self._map_dtype(dtype) or self.torch.float32
        return self.torch.linspace(
            start, stop, num, dtype=resolved, device="cuda"
        )

    def meshgrid(self, *arrays: Array, indexing: str = "xy") -> list[Array]:
        return list(self.torch.meshgrid(*arrays, indexing=indexing))

    def _map_dtype(self, dtype: Any | None) -> Any | None:
        if dtype is None:
            return None
        if isinstance(dtype, str):
            dtype_map = {
                "float32": self.torch.float32,
                "float16": self.torch.float16,
                "bfloat16": self.torch.bfloat16,
                "float64": self.torch.float64,
                "int32": self.torch.int32,
                "int64": self.torch.int64,
                "int16": self.torch.int16,
                "int8": self.torch.int8,
                "uint8": self.torch.uint8,
                "bool": self.torch.bool,
            }
            return dtype_map.get(dtype, dtype)
        name = getattr(dtype, "name", None) or getattr(dtype, "__name__", None) or str(dtype)
        name = name.replace("torch.", "")
        dtype_map = {
            "float32": self.torch.float32,
            "float16": self.torch.float16,
            "bfloat16": self.torch.bfloat16,
            "float64": self.torch.float64,
            "int32": self.torch.int32,
            "int64": self.torch.int64,
            "int16": self.torch.int16,
            "int8": self.torch.int8,
            "uint8": self.torch.uint8,
            "bool": self.torch.bool,
        }
        return dtype_map.get(name, dtype)

    # --- Shape Manipulation (new) ---
    def stack(self, arrays: list[Array], axis: int = 0) -> Array:
        return self.torch.stack(arrays, dim=axis)

    def concatenate(self, arrays: list[Array], axis: int = 0) -> Array:
        return self.torch.cat(arrays, dim=axis)

    def broadcast_to(self, array: Array, shape: tuple[int, ...]) -> Array:
        return array.broadcast_to(shape)

    def tile(self, array: Array, reps: tuple[int, ...] | int) -> Array:
        if isinstance(reps, int):
            reps = (reps,)
        return array.repeat(*reps)

    # --- Reductions (new) ---
    def mean(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        if axis is None:
            return array.mean()
        return array.mean(dim=axis, keepdim=keepdims)

    def min(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        if axis is None:
            return array.min()
        return array.min(dim=axis, keepdim=keepdims).values

    def argmax(self, array: Array, axis: int | None = None) -> Array:
        return array.argmax(dim=axis)

    def argmin(self, array: Array, axis: int | None = None) -> Array:
        return array.argmin(dim=axis)

    def var(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        if axis is None:
            return array.var()
        return array.var(dim=axis, keepdim=keepdims)

    def std(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        if axis is None:
            return array.std()
        return array.std(dim=axis, keepdim=keepdims)

    def all(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        if axis is None:
            return self.torch.all(array)
        return self.torch.all(array, dim=axis, keepdim=keepdims)

    def any(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        if axis is None:
            return self.torch.any(array)
        return self.torch.any(array, dim=axis, keepdim=keepdims)

    # --- Element-wise Operations (new) ---
    def sign(self, array: Array) -> Array:
        return array.sign()

    def isnan(self, array: Array) -> Array:
        return self.torch.isnan(array)

    def isinf(self, array: Array) -> Array:
        return self.torch.isinf(array)

    def isfinite(self, array: Array) -> Array:
        return self.torch.isfinite(array)

    def clip(
        self, array: Array, min_val: float | Array | None, max_val: float | Array | None
    ) -> Array:
        return self.torch.clamp(array, min=min_val, max=max_val)

    def where(self, condition: Array, x: Array, y: Array) -> Array:
        return self.torch.where(condition, x, y)

    def softmax(self, array: Array, axis: int = -1) -> Array:
        return self.torch.softmax(array, dim=axis)

    def cumsum(self, array: Array, axis: int | None = None) -> Array:
        if axis is None:
            return array.flatten().cumsum(dim=0)
        return array.cumsum(dim=axis)

    def floor(self, array: Array) -> Array:
        return self.torch.floor(array)

    def ceil(self, array: Array) -> Array:
        return self.torch.ceil(array)

    def log2(self, array: Array) -> Array:
        return self.torch.log2(array)

    def mod(self, lhs: Array, rhs: Array | float | int) -> Array:
        return self.torch.remainder(lhs, rhs)

    def sin(self, array: Array) -> Array:
        return self.torch.sin(array)

    def cos(self, array: Array) -> Array:
        return self.torch.cos(array)

    def arccos(self, array: Array) -> Array:
        return self.torch.arccos(array)

    def arctan(self, array: Array) -> Array:
        return self.torch.arctan(array)

    def lgamma(self, array: Array) -> Array:
        return self.torch.lgamma(array)

    # --- Linear Algebra (new) ---
    def dot(self, a: Array, b: Array) -> Array:
        if a.ndim == 1 and b.ndim == 1:
            return self.torch.dot(a, b)
        return a @ b

    def norm(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.torch.linalg.norm(array, dim=axis, keepdim=keepdims)

    def det(self, array: Array) -> Array:
        return self.torch.linalg.det(array)

    def linalg_det(self, array: Array) -> Array:
        """Alias for det() for compatibility."""
        return self.det(array)

    def eigh(self, array: Array) -> tuple[Array, Array]:
        return self.torch.linalg.eigh(array)

    def eigvalsh(self, array: Array) -> Array:
        """Compute eigenvalues of symmetric/Hermitian matrix (values only, more efficient)."""
        return self.torch.linalg.eigvalsh(array)

    def solve(self, a: Array, b: Array) -> Array:
        return self.torch.linalg.solve(a, b)

    def inv(self, array: Array) -> Array:
        return self.torch.linalg.inv(array)

    def pinv(self, array: Array) -> Array:
        return self.torch.linalg.pinv(array)

    def cholesky(self, array: Array) -> Array:
        return self.torch.linalg.cholesky(array)

    def trace(self, array: Array) -> Array:
        return self.torch.trace(array)

    def qr(self, array: Array) -> tuple[Array, Array]:
        return self.torch.linalg.qr(array)

    def matrix_sqrt_newton_schulz(self, A: Array, num_iters: int = 15) -> Array:
        """Compute matrix square root via Newton-Schulz iteration.

        Converges to A^{1/2} for positive semi-definite A.
        Runs entirely on GPU.

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
        A_f32 = A.to(dtype=self.torch.float32)

        # Scaling to ensure convergence (spectral norm <= 1)
        padding = self.torch.tensor(1e-7, dtype=self.torch.float32, device="cuda")
        normA_val = self.torch.sqrt(self.torch.sum(A_f32 * A_f32)) + padding
        Y = A_f32 / normA_val

        shape = A.shape
        I = self.torch.eye(shape[0], dtype=self.torch.float32, device="cuda")
        Z = I

        three = self.torch.tensor(3.0, dtype=self.torch.float32, device="cuda")
        half = self.torch.tensor(0.5, dtype=self.torch.float32, device="cuda")

        # Iteration
        for _ in range(num_iters):
            ZY = self.torch.matmul(Z, Y)
            T = three * I - ZY

            Y_new = half * self.torch.matmul(Y, T)
            Z_new = half * self.torch.matmul(T, Z)

            Y = Y_new
            Z = Z_new

        # Rescale
        sqrtA = Y * self.torch.sqrt(normA_val)

        # Cast back if necessary
        if A.dtype != self.torch.float32:
            sqrtA = sqrtA.to(dtype=A.dtype)

        return sqrtA

    def floyd_warshall(self, dist: Array) -> Array:
        """Compute all-pairs shortest paths using Floyd-Warshall on device."""
        dist_arr = dist if hasattr(dist, "device") else self._tensor(dist)
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
                    out = self.torch.minimum(out, via)
                return out

            try:
                compiled = self.compile(_fw)
            except Exception:
                compiled = _fw
            self._compiled_cache[cache_key] = compiled

        return compiled(dist_arr)

    def single_source_shortest_paths(self, dist: Array, source_index: int) -> Array:
        """Compute shortest paths from a single source using Dijkstra-style relaxation."""
        dist_arr = dist if hasattr(dist, "device") else self._tensor(dist)
        if dist_arr.ndim != 2 or dist_arr.shape[0] != dist_arr.shape[1]:
            raise ValueError("single_source_shortest_paths requires a square [n, n] matrix")
        n = int(dist_arr.shape[0])
        if n <= 1:
            return dist_arr[0] if n == 1 else dist_arr

        src = int(source_index)
        if src < 0 or src >= n:
            raise ValueError("source_index out of bounds")

        device = dist_arr.device
        idx = self.torch.arange(n, device=device)
        dist_vec = dist_arr[src]
        visited = (idx == src).to(dist_arr.dtype)
        inf_val = self.torch.tensor(self.torch.finfo(dist_arr.dtype).max, device=device)
        one = self.torch.tensor(1.0, dtype=dist_arr.dtype, device=device)

        for _ in range(n - 1):
            masked = dist_vec + visited * inf_val
            min_idx = self.torch.argmin(masked)
            is_min = idx == min_idx
            visited = self.torch.minimum(visited + is_min.to(dist_arr.dtype), one)
            row = dist_arr[min_idx]
            dist_at_min = dist_vec[min_idx]
            alt = dist_at_min + row
            dist_vec = self.torch.minimum(dist_vec, alt)

        return dist_vec

    # --- Indexing ---
    def take(self, array: Array, indices: Array, axis: int | None = None) -> Array:
        if axis is None:
            return array.flatten()[indices]
        return self.torch.index_select(array, dim=axis, index=indices)

    def take_along_axis(self, array: Array, indices: Array, axis: int) -> Array:
        indices_long = indices.long()
        return self.torch.gather(array, dim=axis, index=indices_long)

    def put_along_axis(
        self, array: Array, indices: Array, values: Array, axis: int | None = None
    ) -> Array:
        if axis is None:
            flat = array.flatten()
            idx = indices.long().flatten()
            vals = values.flatten()
            out = flat.clone()
            out[idx] = vals
            return out.reshape(array.shape)
        indices_long = indices.long()
        return array.scatter(dim=axis, index=indices_long, src=values)

    # --- Sorting ---
    def sort(self, array: Array, axis: int = -1) -> Array:
        return self.torch.sort(array, dim=axis).values

    def argsort(self, array: Array, axis: int = -1) -> Array:
        return self.torch.argsort(array, dim=axis)

    def argpartition(self, array: Array, kth: int, axis: int = -1) -> Array:
        # Exact fallback: full argsort provides a valid partition.
        return self.torch.argsort(array, dim=axis)

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
        # Exact fallback: full sort provides a valid partition.
        return self.torch.sort(array, dim=axis).values

    # --- Random (new) ---
    def random_normal(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        return self.torch.randn(shape, dtype=dtype or self.torch.float32, device="cuda")

    def random_uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        shape: tuple[int, ...] | None = None,
        dtype: Any | None = None,
    ) -> Array:
        shape = shape or (1,)
        return (
            self.torch.rand(shape, dtype=dtype or self.torch.float32, device="cuda") * (high - low)
            + low
        )

    def random_randint(self, low: int, high: int, shape: tuple[int, ...] | None = None) -> Array:
        shape = shape or (1,)
        return self.torch.randint(low, high, shape, device="cuda")

    def random_seed(self, seed: int) -> None:
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed(seed)

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
        mask = self.torch.triu(
            self.torch.full(
                (seq_len, seq_len),
                float("-inf"),
                dtype=dtype or self.torch.float32,
                device="cuda",
            ),
            diagonal=1,
        )
        return mask

    def random_categorical(self, logits: Array, num_samples: int = 1) -> Array:
        """Sample from categorical distribution defined by logits.

        Samples indices from a categorical distribution parameterized by
        unnormalized log-probabilities (logits).

        Args:
            logits: Tensor of shape (..., num_categories) containing logits.
                Can be 1D (single distribution) or 2D (batch of distributions).
            num_samples: Number of samples to draw per distribution.

        Returns:
            Tensor of sampled indices. Shape depends on input:
            - 1D logits: shape (num_samples,)
            - 2D logits (batch_size, num_categories): shape (batch_size, num_samples)
        """
        if logits.dim() == 1:
            probs = self.torch.softmax(logits.unsqueeze(0), dim=-1)
            samples = self.torch.multinomial(probs, num_samples=num_samples, replacement=True)
            return samples.squeeze(0)
        probs = self.torch.softmax(logits, dim=-1)
        return self.torch.multinomial(probs, num_samples=num_samples, replacement=True)

    # =========================================================================
    # SOTA PERFORMANCE APIs (PyTorch 2.x)
    # =========================================================================

    def expand_dims(self, array: Array, axis: int | tuple[int, ...]) -> Array:
        """Add dimension(s) at specified axis position(s).

        Parameters
        ----------
        array : Array
            Input tensor.
        axis : int or tuple of int
            Position(s) where new axes should be inserted.

        Returns
        -------
        Array
            Tensor with expanded dimensions.
        """
        if isinstance(axis, tuple):
            result = array
            for ax in sorted(axis):
                result = result.unsqueeze(dim=ax)
            return result
        return array.unsqueeze(dim=axis)

    def clear_cache(self) -> None:
        """Clear CUDA memory cache."""
        self.torch.cuda.empty_cache()

    def compile(
        self,
        fun: Callable,
        inputs: list | None = None,
        outputs: list | None = None,
        shapeless: bool = False,
    ) -> Callable:
        """JIT-compile a function using torch.compile (TorchInductor).

        Parameters
        ----------
        fun : Callable
            Function to compile.
        inputs : list, optional
            Unused, kept for API compatibility with MLX.
        outputs : list, optional
            Unused, kept for API compatibility with MLX.
        shapeless : bool, optional
            If True, use dynamic shapes. Default is False.

        Returns
        -------
        Callable
            Compiled function with optimized kernels.
        """
        return self.torch.compile(fun, dynamic=shapeless)

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
        return self.torch.vmap(fun, in_dims=in_axes, out_dims=out_axes)

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
        def value_and_grad_fn(*args):
            # Enable gradients for specified arguments
            if isinstance(argnums, int):
                argnum_list = [argnums]
            else:
                argnum_list = list(argnums)

            # Clone and enable gradients for specified args
            new_args = list(args)
            for i in argnum_list:
                if i < len(new_args) and hasattr(new_args[i], 'requires_grad_'):
                    new_args[i] = new_args[i].clone().requires_grad_(True)

            # Forward pass
            value = fun(*new_args)

            # Backward pass
            value.backward()

            # Collect gradients
            if isinstance(argnums, int):
                grad = new_args[argnums].grad
            else:
                grad = tuple(new_args[i].grad for i in argnum_list)

            return value.detach(), grad

        return value_and_grad_fn

    def async_eval(self, *arrays: Array) -> None:
        """Asynchronously evaluate arrays without blocking.

        Parameters
        ----------
        *arrays : Array
            Arrays to evaluate asynchronously.

        Notes
        -----
        CUDA operations are asynchronous by default. This is a no-op
        for API compatibility.
        """
        # CUDA is async by default - operations are queued and executed
        # asynchronously. No explicit action needed.
        return None

    # --- Fused CUDA Kernels ---

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
            Input tensor to normalize.
        weight : Array or None
            Scaling weights. If None, returns normalized input.
        eps : float, optional
            Epsilon for numerical stability. Default is 1e-5.
        stream : Any, optional
            Unused in CUDA backend. Included for API compatibility.

        Returns
        -------
        Array
            RMS-normalized output.
        """
        # PyTorch 2.5+ has native rms_norm, fallback for older versions
        if hasattr(self.torch.nn.functional, "rms_norm"):
            normalized_shape = (x.shape[-1],)
            return self.torch.nn.functional.rms_norm(x, normalized_shape, weight, eps)
        # Manual implementation for older PyTorch
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * self.torch.rsqrt(variance + eps)
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
        """Apply layer normalization using fused kernel.

        Parameters
        ----------
        x : Array
            Input tensor to normalize.
        weight : Array or None
            Scaling weights.
        bias : Array or None
            Bias terms.
        eps : float, optional
            Epsilon for numerical stability. Default is 1e-5.
        stream : Any, optional
            Unused in CUDA backend. Included for API compatibility.

        Returns
        -------
        Array
            Layer-normalized output.
        """
        normalized_shape = (x.shape[-1],)
        return self.torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

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
            Input tensor of shape (..., seq_len, dims).
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
            Unused in CUDA backend. Included for API compatibility.

        Returns
        -------
        Array
            Output with rotary position embeddings applied.
        """
        # RoPE implementation following standard formulation
        seq_len = x.shape[-2]
        half_dims = dims // 2

        if freqs is None:
            if base is None:
                raise ValueError("rope() expects base when freqs is not provided")
            inv_freq = 1.0 / (
                base
                ** (
                    self.torch.arange(0, half_dims, dtype=x.dtype, device=x.device)
                    / half_dims
                )
            )
            positions = self.torch.arange(seq_len, device=x.device, dtype=x.dtype)
            if isinstance(offset, (int, float)):
                positions = positions + offset
            else:
                offset_arr = self.torch.as_tensor(offset, dtype=x.dtype, device=x.device)
                positions = positions + offset_arr[..., None]
            positions = positions * scale
            freqs = positions[..., None] * inv_freq
        else:
            if base is not None:
                raise ValueError("rope() expects base=None when freqs is provided")
            freqs = self.torch.as_tensor(freqs, dtype=x.dtype, device=x.device)

        cos = freqs.cos()
        sin = freqs.sin()

        # Reshape for broadcasting
        if cos.ndim == 2:
            cos = cos.view((1,) * (x.ndim - 2) + cos.shape)
            sin = sin.view((1,) * (x.ndim - 2) + sin.shape)
        elif cos.ndim == 3:
            cos = cos.view((cos.shape[0],) + (1,) * (x.ndim - 3) + cos.shape[1:])
            sin = sin.view((sin.shape[0],) + (1,) * (x.ndim - 3) + sin.shape[1:])

        # Split x into two halves
        x1 = x[..., :half_dims]
        x2 = x[..., half_dims:dims]

        if traditional:
            # Traditional RoPE: interleaved rotation
            rotated = self.torch.cat([-x2, x1], dim=-1)
            x_rope = x[..., :dims] * cos.repeat(1, 1, 1, 2) + rotated * sin.repeat(1, 1, 1, 2)
        else:
            # Modern RoPE: paired rotation
            x_rope = self.torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        # Preserve dimensions beyond RoPE range
        if x.shape[-1] > dims:
            return self.torch.cat([x_rope, x[..., dims:]], dim=-1)
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
        """Compute scaled dot-product attention using FlashAttention.

        Parameters
        ----------
        q : Array
            Query tensor.
        k : Array
            Key tensor.
        v : Array
            Value tensor.
        scale : float
            Scaling factor for attention scores.
        mask : Array, str, or None, optional
            Attention mask. Use "causal" for causal masking.
        sinks : Array or None, optional
            Attention sinks. Not supported in CUDA backend.
        stream : Any, optional
            Unused in CUDA backend. Included for API compatibility.

        Returns
        -------
        Array
            Attention output.
        """
        if sinks is not None:
            scores = self.torch.einsum("...qhd,...khd->...hqk", q, k) * scale
            if mask is not None:
                if isinstance(mask, str):
                    if mask != "causal":
                        raise ValueError(f"Unsupported attention mask: {mask}")
                    t_q = q.shape[-2]
                    t_kv = k.shape[-2]
                    mask = self.torch.tril(
                        self.torch.ones((t_q, t_kv), dtype=self.torch.bool, device=q.device)
                    )
                mask_arr = self.torch.as_tensor(mask, device=q.device)
                if mask_arr.dtype == self.torch.bool:
                    neg_inf = self.torch.finfo(scores.dtype).min
                    scores = self.torch.where(mask_arr, scores, neg_inf)
                else:
                    scores = scores + mask_arr
            sinks_arr = self.torch.as_tensor(sinks, device=q.device)
            scores = scores + sinks_arr
            attn_weights = self.torch.softmax(scores, dim=-1)
            return self.torch.einsum("...hqk,...khd->...qhd", attn_weights, v)

        is_causal = False
        attn_mask = mask
        if isinstance(mask, str):
            if mask != "causal":
                raise ValueError(f"Unsupported attention mask: {mask}")
            attn_mask = None
            is_causal = True
        return self.torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
        )

    # --- Stream Management for CPU/GPU Parallelism ---

    def new_stream(self, device: str = "gpu") -> Any:
        """Create a new CUDA stream for parallel execution.

        Args:
            device: "gpu" or "cpu" (cpu returns None)

        Returns:
            CUDA Stream object for parallel execution
        """
        if device == "cpu":
            return None
        return self.torch.cuda.Stream()

    def synchronize(self) -> None:
        """Synchronize all CUDA streams (wait for all GPU work to complete)."""
        self.torch.cuda.synchronize()

    # --- File I/O (Native Backend Serialization) ---

    def save_safetensors(
        self, path: str, weights: dict[str, Any], metadata: dict[str, str] | None = None
    ) -> None:
        """Save weights to safetensors using PyTorch native I/O.

        Args:
            path: File path to save to.
            weights: Dictionary of weight name -> array.
            metadata: Optional dictionary of string metadata to include.
        """
        from safetensors.torch import save_file

        # Ensure all arrays are torch tensors
        torch_weights = {}
        for key, value in weights.items():
            if isinstance(value, self.torch.Tensor):
                torch_weights[key] = value
            else:
                torch_weights[key] = self.array(value)
        if metadata:
            save_file(torch_weights, path, metadata=metadata)
        else:
            save_file(torch_weights, path)

    def load_safetensors(self, path: str) -> dict[str, Any]:
        """Load weights from safetensors using PyTorch native I/O.

        Args:
            path: File path to load from.

        Returns:
            Dictionary of weight name -> torch tensor on CUDA.
        """
        from safetensors.torch import load_file

        weights = load_file(path, device="cuda")
        return weights
