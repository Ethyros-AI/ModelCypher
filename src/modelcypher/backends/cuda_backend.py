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

from modelcypher.ports.backend import Array, Backend, FloatInfo


class CUDABackend(Backend):
    def __init__(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for the CUDA backend") from exc
        self.torch = torch

    def _tensor(self, data: Any, dtype: Any | None = None) -> Array:
        dtype = dtype or self.torch.float32
        return self.torch.tensor(data, dtype=dtype, device="cuda")

    def array(self, data: Any, dtype: Any | None = None) -> Array:
        return self._tensor(data, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        return self.torch.zeros(shape, dtype=dtype or self.torch.float32, device="cuda")

    def ones(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        return self.torch.ones(shape, dtype=dtype or self.torch.float32, device="cuda")

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

    def abs(self, array: Array) -> Array:
        return array.abs()

    def astype(self, array: Array, dtype: Any) -> Array:
        return array.to(dtype)

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
        raise NotImplementedError("Quantized weights are not supported on the CUDA backend.")

    def dequantize(
        self,
        weight: Array,
        scales: Array,
        biases: Array | None,
        group_size: int,
        bits: int,
        mode: str,
    ) -> Array:
        raise NotImplementedError("Quantized weights are not supported on the CUDA backend.")

    def eval(self, *arrays: Array) -> None:
        self.torch.cuda.synchronize()

    def to_numpy(self, array: Array) -> Any:
        return array.detach().cpu().numpy()

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
        return self.torch.eye(n, m or n, dtype=dtype or self.torch.float32, device="cuda")

    def arange(
        self,
        start: int | float,
        stop: int | float | None = None,
        step: int | float = 1,
        dtype: Any | None = None,
    ) -> Array:
        if stop is None:
            return self.torch.arange(start, dtype=dtype, device="cuda")
        return self.torch.arange(start, stop, step, dtype=dtype, device="cuda")

    def diag(self, array: Array, k: int = 0) -> Array:
        return self.torch.diag(array, diagonal=k)

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: Any | None = None) -> Array:
        return self.torch.full(shape, fill_value, dtype=dtype or self.torch.float32, device="cuda")

    def ones_like(self, array: Array, dtype: Any | None = None) -> Array:
        return self.torch.ones_like(array, dtype=dtype)

    def zeros_like(self, array: Array, dtype: Any | None = None) -> Array:
        return self.torch.zeros_like(array, dtype=dtype)

    def linspace(self, start: float, stop: float, num: int, dtype: Any | None = None) -> Array:
        return self.torch.linspace(
            start, stop, num, dtype=dtype or self.torch.float32, device="cuda"
        )

    # --- Shape Manipulation (new) ---
    def stack(self, arrays: list[Array], axis: int = 0) -> Array:
        return self.torch.stack(arrays, dim=axis)

    def concatenate(self, arrays: list[Array], axis: int = 0) -> Array:
        return self.torch.cat(arrays, dim=axis)

    def broadcast_to(self, array: Array, shape: tuple[int, ...]) -> Array:
        return array.broadcast_to(shape)

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

    # --- Element-wise Operations (new) ---
    def sign(self, array: Array) -> Array:
        return array.sign()

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

    def eigh(self, array: Array) -> tuple[Array, Array]:
        return self.torch.linalg.eigh(array)

    def solve(self, a: Array, b: Array) -> Array:
        return self.torch.linalg.solve(a, b)

    def inv(self, array: Array) -> Array:
        return self.torch.linalg.inv(array)

    def cholesky(self, array: Array) -> Array:
        return self.torch.linalg.cholesky(array)

    def trace(self, array: Array) -> Array:
        return self.torch.trace(array)

    def qr(self, array: Array) -> tuple[Array, Array]:
        return self.torch.linalg.qr(array)

    # --- Indexing ---
    def take(self, array: Array, indices: Array, axis: int | None = None) -> Array:
        if axis is None:
            return array.flatten()[indices]
        return self.torch.index_select(array, dim=axis, index=indices)

    # --- Sorting ---
    def sort(self, array: Array, axis: int = -1) -> Array:
        return self.torch.sort(array, dim=axis).values

    def argsort(self, array: Array, axis: int = -1) -> Array:
        return self.torch.argsort(array, dim=axis)

    def argpartition(self, array: Array, kth: int, axis: int = -1) -> Array:
        # PyTorch doesn't have argpartition directly; use topk as approximation
        # topk returns smallest k+1 elements when largest=False
        _, indices = self.torch.topk(array, k=kth + 1, dim=axis, largest=False)
        return indices

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
        # PyTorch lacks native partition; use topk for approximation
        n = array.shape[axis]
        if kth >= n:
            return self.torch.sort(array, dim=axis).values

        # Get bottom k+1 elements (values up to and including kth position)
        bottom_k, _ = self.torch.topk(array, k=kth + 1, dim=axis, largest=False)
        # Get top (n - k - 1) elements
        top_rest, _ = self.torch.topk(array, k=n - kth - 1, dim=axis, largest=True)
        # Concatenate: [smallest k] + [kth pivot] + [largest n-k-1]
        return self.torch.cat([bottom_k, top_rest], dim=axis)

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
        pass

    # --- Fused CUDA Kernels ---

    def rms_norm(self, x: Array, weight: Array, eps: float = 1e-5) -> Array:
        """Apply RMS normalization using fused kernel.

        Parameters
        ----------
        x : Array
            Input tensor to normalize.
        weight : Array
            Scaling weights.
        eps : float, optional
            Epsilon for numerical stability. Default is 1e-5.

        Returns
        -------
        Array
            RMS-normalized output.
        """
        # PyTorch 2.5+ has native rms_norm, fallback for older versions
        if hasattr(self.torch.nn.functional, "rms_norm"):
            return self.torch.nn.functional.rms_norm(x, weight.shape, weight, eps)
        # Manual implementation for older PyTorch
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * self.torch.rsqrt(variance + eps)
        return x_normed * weight

    def layer_norm(
        self, x: Array, weight: Array | None, bias: Array | None, eps: float = 1e-5
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
        base: float = 10000.0,
        scale: float = 1.0,
        offset: int = 0,
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
        # RoPE implementation following standard formulation
        seq_len = x.shape[-2]
        half_dims = dims // 2

        # Compute frequencies
        inv_freq = 1.0 / (
            base ** (self.torch.arange(0, half_dims, dtype=x.dtype, device=x.device) / half_dims)
        )

        # Position indices
        positions = (self.torch.arange(seq_len, device=x.device) + offset) * scale

        # Compute sin/cos: [seq_len, half_dims]
        freqs = self.torch.einsum("i,j->ij", positions, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()

        # Reshape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, half_dims]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, half_dims]

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
        mask: Array | None = None,
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
        mask : Array or None, optional
            Attention mask. Default is None.

        Returns
        -------
        Array
            Attention output.
        """
        return self.torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale
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
