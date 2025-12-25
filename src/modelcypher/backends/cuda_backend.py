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

from typing import Any

from modelcypher.ports.backend import Array, Backend


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
        """Approximate O(n) partition via torch.kthvalue + topk.

        PyTorch has torch.kthvalue for finding kth element, but no full partition.
        We use topk for the smaller portion and kthvalue for the pivot.
        """
        # For a true partition: elements < kth are smaller, elements > kth are larger
        # torch.kthvalue gives us the pivot value and we can use topk for efficiency
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
