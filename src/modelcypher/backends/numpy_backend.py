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

import math
from typing import Any, Callable

import numpy as np

from modelcypher.ports.backend import Array, Backend, FloatInfo


class NumpyBackend(Backend):
    """CPU backend for ModelCypher using NumPy vectorized operations."""

    def __init__(self) -> None:
        self.np = np
        self._rng = np.random.default_rng(0)

    def _map_dtype(self, dtype: Any | None) -> Any | None:
        if dtype is None:
            return None
        if isinstance(dtype, str):
            normalized = dtype.replace("numpy.", "").replace("np.", "")
            if normalized.startswith("mlx.core."):
                normalized = normalized.replace("mlx.core.", "")
            if normalized == "bfloat16":
                return self.np.float32
            try:
                return self.np.dtype(normalized)
            except TypeError:
                return dtype
        if isinstance(dtype, self.np.dtype):
            return dtype
        if dtype is self.np.float16:
            return self.np.float16
        if dtype is self.np.float32:
            return self.np.float32
        if dtype is self.np.float64:
            return self.np.float64
        if dtype is self.np.int32:
            return self.np.int32
        if dtype is self.np.int64:
            return self.np.int64
        if dtype is self.np.int16:
            return self.np.int16
        if dtype is self.np.int8:
            return self.np.int8
        if dtype is self.np.uint8:
            return self.np.uint8
        if dtype is self.np.bool_:
            return self.np.bool_
        return dtype

    # --- Array Creation ---
    def array(self, data: Any, dtype: Any | None = None) -> Array:
        return self.np.array(data, dtype=self._map_dtype(dtype))

    def zeros(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        return self.np.zeros(shape, dtype=self._map_dtype(dtype) or self.np.float32)

    def ones(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        return self.np.ones(shape, dtype=self._map_dtype(dtype) or self.np.float32)

    def eye(self, n: int, m: int | None = None, dtype: Any | None = None) -> Array:
        return self.np.eye(n, m or n, dtype=self._map_dtype(dtype) or self.np.float32)

    def arange(
        self,
        start: int | float,
        stop: int | float | None = None,
        step: int | float = 1,
        dtype: Any | None = None,
    ) -> Array:
        if stop is None:
            return self.np.arange(start, dtype=self._map_dtype(dtype) or self.np.float32)
        return self.np.arange(start, stop, step, dtype=self._map_dtype(dtype) or self.np.float32)

    def diag(self, array: Array, k: int = 0) -> Array:
        return self.np.diag(array, k=k)

    def dtype(self, array: Array) -> Any:
        return array.dtype

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: Any | None = None) -> Array:
        return self.np.full(shape, fill_value, dtype=self._map_dtype(dtype) or self.np.float32)

    def ones_like(self, array: Array, dtype: Any | None = None) -> Array:
        return self.np.ones_like(array, dtype=self._map_dtype(dtype))

    def zeros_like(self, array: Array, dtype: Any | None = None) -> Array:
        return self.np.zeros_like(array, dtype=self._map_dtype(dtype))

    def linspace(self, start: float, stop: float, num: int, dtype: Any | None = None) -> Array:
        return self.np.linspace(start, stop, num, dtype=self._map_dtype(dtype) or self.np.float32)

    # --- Shape Manipulation ---
    def shape(self, array: Array) -> tuple[int, ...]:
        return tuple(array.shape)

    def reshape(self, array: Array, shape: tuple[int, ...]) -> Array:
        return self.np.reshape(array, shape)

    def squeeze(self, array: Array, axis: int | None = None) -> Array:
        return self.np.squeeze(array, axis=axis)

    def transpose(self, array: Array, axes: tuple[int, ...] | None = None) -> Array:
        return self.np.transpose(array, axes=axes)

    def stack(self, arrays: list[Array], axis: int = 0) -> Array:
        return self.np.stack(arrays, axis=axis)

    def concatenate(self, arrays: list[Array], axis: int = 0) -> Array:
        return self.np.concatenate(arrays, axis=axis)

    def broadcast_to(self, array: Array, shape: tuple[int, ...]) -> Array:
        return self.np.broadcast_to(array, shape)

    def expand_dims(self, array: Array, axis: int | tuple[int, ...]) -> Array:
        return self.np.expand_dims(array, axis=axis)

    # --- Reductions ---
    def sum(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.np.sum(array, axis=axis, keepdims=keepdims)

    def mean(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.np.mean(array, axis=axis, keepdims=keepdims)

    def max(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return self.np.max(array, axis=axis, keepdims=keepdims)

    def min(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return self.np.min(array, axis=axis, keepdims=keepdims)

    def argmax(self, array: Array, axis: int | None = None) -> Array:
        return self.np.argmax(array, axis=axis)

    def argmin(self, array: Array, axis: int | None = None) -> Array:
        return self.np.argmin(array, axis=axis)

    def var(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.np.var(array, axis=axis, keepdims=keepdims)

    def std(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.np.std(array, axis=axis, keepdims=keepdims)

    # --- Element-wise Operations ---
    def sqrt(self, array: Array) -> Array:
        return self.np.sqrt(array)

    def exp(self, array: Array) -> Array:
        return self.np.exp(array)

    def log(self, array: Array) -> Array:
        return self.np.log(array)

    def abs(self, array: Array) -> Array:
        return self.np.abs(array)

    def sign(self, array: Array) -> Array:
        return self.np.sign(array)

    def isnan(self, array: Array) -> Array:
        return self.np.isnan(array)

    def isinf(self, array: Array) -> Array:
        return self.np.isinf(array)

    def isfinite(self, array: Array) -> Array:
        return self.np.isfinite(array)

    def sin(self, array: Array) -> Array:
        return self.np.sin(array)

    def cos(self, array: Array) -> Array:
        return self.np.cos(array)

    def arccos(self, array: Array) -> Array:
        return self.np.arccos(array)

    def arctan(self, array: Array) -> Array:
        return self.np.arctan(array)

    def lgamma(self, array: Array) -> Array:
        vectorized = self.np.vectorize(math.lgamma)
        return vectorized(array)

    def maximum(self, lhs: Array, rhs: Array) -> Array:
        return self.np.maximum(lhs, rhs)

    def minimum(self, lhs: Array, rhs: Array) -> Array:
        return self.np.minimum(lhs, rhs)

    def clip(
        self, array: Array, min_val: float | Array | None, max_val: float | Array | None
    ) -> Array:
        return self.np.clip(array, min_val, max_val)

    def where(self, condition: Array, x: Array, y: Array) -> Array:
        return self.np.where(condition, x, y)

    def softmax(self, array: Array, axis: int = -1) -> Array:
        shifted = array - self.np.max(array, axis=axis, keepdims=True)
        exp_vals = self.np.exp(shifted)
        return exp_vals / self.np.sum(exp_vals, axis=axis, keepdims=True)

    def cumsum(self, array: Array, axis: int | None = None) -> Array:
        return self.np.cumsum(array, axis=axis)

    def floor(self, array: Array) -> Array:
        return self.np.floor(array)

    def ceil(self, array: Array) -> Array:
        return self.np.ceil(array)

    def log2(self, array: Array) -> Array:
        return self.np.log2(array)

    # --- Linear Algebra ---
    def matmul(self, lhs: Array, rhs: Array) -> Array:
        return self.np.matmul(lhs, rhs)

    def dot(self, a: Array, b: Array) -> Array:
        return self.np.dot(a, b)

    def svd(
        self,
        array: Array,
        compute_uv: bool = True,
        full_matrices: bool | None = None,
    ) -> tuple[Array, Array, Array] | Array:
        use_full = bool(full_matrices) if full_matrices is not None else False
        if compute_uv:
            u, s, vt = self.np.linalg.svd(array, full_matrices=use_full, compute_uv=True)
            return u, s, vt
        return self.np.linalg.svd(array, full_matrices=use_full, compute_uv=False)

    def norm(
        self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return self.np.linalg.norm(array, axis=axis, keepdims=keepdims)

    def det(self, array: Array) -> Array:
        return self.np.linalg.det(array)

    def linalg_det(self, array: Array) -> Array:
        return self.det(array)

    def eigh(self, array: Array) -> tuple[Array, Array]:
        eigenvalues, eigenvectors = self.np.linalg.eigh(array)
        return eigenvalues, eigenvectors

    def solve(self, a: Array, b: Array) -> Array:
        return self.np.linalg.solve(a, b)

    def inv(self, array: Array) -> Array:
        return self.np.linalg.inv(array)

    def pinv(self, array: Array) -> Array:
        return self.np.linalg.pinv(array)

    def cholesky(self, array: Array) -> Array:
        return self.np.linalg.cholesky(array)

    def trace(self, array: Array) -> Array:
        return self.np.trace(array)

    def qr(self, array: Array) -> tuple[Array, Array]:
        q, r = self.np.linalg.qr(array)
        return q, r

    # --- Indexing ---
    def take(self, array: Array, indices: Array, axis: int | None = None) -> Array:
        return self.np.take(array, indices, axis=axis)

    # --- Sorting ---
    def sort(self, array: Array, axis: int = -1) -> Array:
        return self.np.sort(array, axis=axis)

    def argsort(self, array: Array, axis: int = -1) -> Array:
        return self.np.argsort(array, axis=axis)

    def argpartition(self, array: Array, kth: int, axis: int = -1) -> Array:
        return self.np.argpartition(array, kth, axis=axis)

    def partition(self, array: Array, kth: int, axis: int = -1) -> Array:
        return self.np.partition(array, kth, axis=axis)

    # --- Random ---
    def random_normal(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        data = self._rng.standard_normal(size=shape)
        mapped = self._map_dtype(dtype) or self.np.float32
        return data.astype(mapped)

    def random_uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        shape: tuple[int, ...] | None = None,
        dtype: Any | None = None,
    ) -> Array:
        data = self._rng.uniform(low=low, high=high, size=shape or (1,))
        mapped = self._map_dtype(dtype) or self.np.float32
        return data.astype(mapped)

    def random_randint(
        self, low: int, high: int, shape: tuple[int, ...] | None = None
    ) -> Array:
        return self._rng.integers(low, high, size=shape or (1,))

    def random_seed(self, seed: int) -> None:
        self._rng = self.np.random.default_rng(seed)

    def random_categorical(self, logits: Array, num_samples: int = 1) -> Array:
        logits_arr = self.np.array(logits)
        shifted = logits_arr - self.np.max(logits_arr, axis=-1, keepdims=True)
        exp_vals = self.np.exp(shifted)
        probs = exp_vals / self.np.sum(exp_vals, axis=-1, keepdims=True)

        if logits_arr.ndim == 1:
            return self._rng.choice(
                logits_arr.shape[0], size=(num_samples,), replace=True, p=probs
            )

        flat = probs.reshape(-1, probs.shape[-1])
        samples = [
            self._rng.choice(flat.shape[1], size=(num_samples,), replace=True, p=row)
            for row in flat
        ]
        result = self.np.stack(samples, axis=0)
        out_shape = logits_arr.shape[:-1] + (num_samples,)
        return result.reshape(out_shape)

    # --- Type Conversion ---
    def astype(self, array: Array, dtype: Any) -> Array:
        return self.np.asarray(array).astype(self._map_dtype(dtype))

    def to_numpy(self, array: Array) -> Any:
        return self.np.asarray(array)

    def to_scalar(self, array: Array) -> float | int:
        arr = self.np.asarray(array)
        if arr.size != 1:
            raise ValueError("Array must contain exactly one element for to_scalar()")
        return arr.item()

    def tolist(self, array: Array) -> list | float | int:
        return self.np.asarray(array).tolist()

    def finfo(self, dtype: Any | None = None) -> FloatInfo:
        resolved = self._map_dtype(dtype) or self.np.float32
        if isinstance(resolved, self.np.dtype) and "bfloat16" in str(resolved):
            resolved = self.np.float32
        info = self.np.finfo(resolved)
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
        weight_arr = self.np.asarray(weight)
        shape = weight_arr.shape
        if len(shape) < 2:
            weight_arr = weight_arr.reshape(-1, 1)

        num_groups = weight_arr.shape[0] // group_size
        weight_grouped = weight_arr.reshape(num_groups, group_size, -1)

        max_vals = self.np.max(self.np.abs(weight_grouped), axis=1, keepdims=True)
        scales = max_vals / (2 ** (bits - 1) - 1)
        scales = self.np.where(scales == 0, 1.0, scales)

        weight_q = self.np.round(weight_grouped / scales)
        weight_q = self.np.clip(weight_q, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        weight_q = weight_q.astype(self.np.int8)

        return weight_q.reshape(shape), scales.reshape(-1, weight_arr.shape[-1]), None

    def dequantize(
        self,
        weight: Array,
        scales: Array,
        biases: Array | None,
        group_size: int,
        bits: int,
        mode: str,
    ) -> Array:
        weight_arr = self.np.asarray(weight).astype(self.np.float32)
        shape = weight_arr.shape
        if len(shape) < 2:
            weight_arr = weight_arr.reshape(-1, 1)

        num_groups = weight_arr.shape[0] // group_size
        weight_grouped = weight_arr.reshape(num_groups, group_size, -1)
        scales_grouped = self.np.asarray(scales).reshape(num_groups, 1, -1)

        dequantized = weight_grouped * scales_grouped
        if biases is not None:
            biases_grouped = self.np.asarray(biases).reshape(num_groups, 1, -1)
            dequantized = dequantized + biases_grouped

        return dequantized.reshape(shape)

    # --- Attention Masks ---
    def create_causal_mask(self, seq_len: int, dtype: Any | None = None) -> Array:
        mapped = self._map_dtype(dtype) or self.np.float32
        mask = self.np.full((seq_len, seq_len), float("-inf"), dtype=mapped)
        return self.np.triu(mask, k=1)

    # --- Compute Control ---
    def eval(self, *arrays: Array) -> None:
        return None

    def clear_cache(self) -> None:
        return None

    # --- Performance APIs ---
    def compile(
        self,
        fun: Callable,
        inputs: list | None = None,
        outputs: list | None = None,
        shapeless: bool = False,
    ) -> Callable:
        return fun

    def vmap(
        self,
        fun: Callable,
        in_axes: int | tuple | None = 0,
        out_axes: int | tuple | None = 0,
    ) -> Callable:
        def _wrapped(*args):
            if in_axes is None:
                return fun(*args)

            if isinstance(in_axes, tuple):
                axes = in_axes
            else:
                axes = tuple(in_axes for _ in args)

            batch_sizes = [
                args[idx].shape[ax]
                for idx, ax in enumerate(axes)
                if ax is not None and hasattr(args[idx], "shape")
            ]
            if not batch_sizes:
                return fun(*args)

            batch = batch_sizes[0]
            outputs = []
            for i in range(batch):
                sliced = []
                for arg, ax in zip(args, axes):
                    if ax is None:
                        sliced.append(arg)
                    else:
                        sliced.append(self.np.take(arg, i, axis=ax))
                outputs.append(fun(*sliced))

            if isinstance(outputs[0], tuple):
                stacked = []
                for out_idx in range(len(outputs[0])):
                    stacked.append(
                        self.np.stack([out[out_idx] for out in outputs], axis=out_axes)
                    )
                return tuple(stacked)
            return self.np.stack(outputs, axis=out_axes)

        return _wrapped

    def async_eval(self, *arrays: Array) -> None:
        return None

    # --- Fused Operations ---
    def rms_norm(self, x: Array, weight: Array, eps: float = 1e-5) -> Array:
        mean_sq = self.np.mean(x * x, axis=-1, keepdims=True)
        inv_rms = 1.0 / self.np.sqrt(mean_sq + eps)
        return x * inv_rms * weight

    def layer_norm(
        self, x: Array, weight: Array | None, bias: Array | None, eps: float = 1e-5
    ) -> Array:
        mean = self.np.mean(x, axis=-1, keepdims=True)
        var = self.np.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
        normalized = (x - mean) / self.np.sqrt(var + eps)
        if weight is not None:
            normalized = normalized * weight
        if bias is not None:
            normalized = normalized + bias
        return normalized

    def rope(
        self,
        x: Array,
        dims: int,
        traditional: bool = False,
        base: float = 10000.0,
        scale: float = 1.0,
        offset: int = 0,
    ) -> Array:
        x_arr = self.np.asarray(x)
        if dims <= 0:
            return x_arr

        last_dim = x_arr.shape[-1]
        rot_dim = min(dims, last_dim)
        if rot_dim % 2 != 0:
            rot_dim -= 1
        if rot_dim <= 0:
            return x_arr

        seq_len = x_arr.shape[-2]
        positions = self.np.arange(seq_len) + offset
        inv_freq = 1.0 / (base ** (self.np.arange(0, rot_dim, 2) / rot_dim))
        freqs = (positions * scale)[:, None] * inv_freq[None, :]

        cos = self.np.cos(freqs)
        sin = self.np.sin(freqs)
        for _ in range(x_arr.ndim - 2):
            cos = cos[None, ...]
            sin = sin[None, ...]

        x_rot = x_arr[..., :rot_dim]
        x1 = x_rot[..., 0::2]
        x2 = x_rot[..., 1::2]
        rotated = self.np.concatenate(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1
        )

        if rot_dim < last_dim:
            return self.np.concatenate([rotated, x_arr[..., rot_dim:]], axis=-1)
        return rotated

    def scaled_dot_product_attention(
        self,
        q: Array,
        k: Array,
        v: Array,
        scale: float,
        mask: Array | None = None,
    ) -> Array:
        scores = self.np.matmul(q, self.np.swapaxes(k, -1, -2)) * scale
        if mask is not None:
            scores = scores + mask
        weights = self.softmax(scores, axis=-1)
        return self.np.matmul(weights, v)

    # --- Stream Management ---
    def new_stream(self, device: str = "gpu") -> Any:
        return None

    def synchronize(self) -> None:
        return None

    # --- File I/O ---
    def save_safetensors(
        self, path: str, weights: dict[str, Array], metadata: dict[str, str] | None = None
    ) -> None:
        from safetensors.numpy import save_file

        weights_np = {key: self.to_numpy(value) for key, value in weights.items()}
        if metadata:
            save_file(weights_np, path, metadata=metadata)
        else:
            save_file(weights_np, path)

    def load_safetensors(self, path: str) -> dict[str, Array]:
        from safetensors.numpy import load_file

        return load_file(path)

