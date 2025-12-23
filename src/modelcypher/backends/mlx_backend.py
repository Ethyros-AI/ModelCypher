from __future__ import annotations

from typing import Any

import numpy as np

from modelcypher.backends.safe_gpu import SafeGPU
from modelcypher.ports.backend import Backend, Array


class MLXBackend(Backend):
    def __init__(self) -> None:
        import mlx.core as mx

        self.mx = mx
        self.safe = SafeGPU(mx)

    def array(self, data: Any, dtype: Any | None = None) -> Array:
        arr = self.mx.array(data, dtype=self._map_dtype(dtype))
        self.safe.eval(arr)
        return arr

    def zeros(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        arr = self.mx.zeros(shape, dtype=self._map_dtype(dtype))
        self.safe.eval(arr)
        return arr

    def ones(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        arr = self.mx.ones(shape, dtype=self._map_dtype(dtype))
        self.safe.eval(arr)
        return arr

    def reshape(self, array: Array, shape: tuple[int, ...]) -> Array:
        arr = self.mx.reshape(array, shape)
        self.safe.eval(arr)
        return arr

    def squeeze(self, array: Array, axis: int | None = None) -> Array:
        arr = self.mx.squeeze(array, axis=axis) if axis is not None else self.mx.squeeze(array)
        self.safe.eval(arr)
        return arr

    def transpose(self, array: Array, axes: tuple[int, ...] | None = None) -> Array:
        arr = self.mx.transpose(array, axes=axes) if axes else self.mx.transpose(array)
        self.safe.eval(arr)
        return arr

    def matmul(self, lhs: Array, rhs: Array) -> Array:
        arr = self.mx.matmul(lhs, rhs)
        self.safe.eval(arr)
        return arr

    def sum(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        arr = self.mx.sum(array, axis=axis, keepdims=keepdims)
        self.safe.eval(arr)
        return arr

    def max(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        arr = self.mx.max(array, axis=axis, keepdims=keepdims)
        self.safe.eval(arr)
        return arr

    def sqrt(self, array: Array) -> Array:
        arr = self.mx.sqrt(array)
        self.safe.eval(arr)
        return arr

    def exp(self, array: Array) -> Array:
        arr = self.mx.exp(array)
        self.safe.eval(arr)
        return arr

    def log(self, array: Array) -> Array:
        arr = self.mx.log(array)
        self.safe.eval(arr)
        return arr

    def maximum(self, lhs: Array, rhs: Array) -> Array:
        arr = self.mx.maximum(lhs, rhs)
        self.safe.eval(arr)
        return arr

    def minimum(self, lhs: Array, rhs: Array) -> Array:
        arr = self.mx.minimum(lhs, rhs)
        self.safe.eval(arr)
        return arr

    def abs(self, array: Array) -> Array:
        arr = self.mx.abs(array)
        self.safe.eval(arr)
        return arr

    def astype(self, array: Array, dtype: Any) -> Array:
        arr = array.astype(self._map_dtype(dtype))
        self.safe.eval(arr)
        return arr

    def svd(self, array: Array, compute_uv: bool = True) -> tuple[Array, Array, Array] | Array:
        result = self.mx.linalg.svd(array, compute_uv=compute_uv)
        if compute_uv:
            u, s, vt = result
            self.safe.eval(u, s, vt)
            return u, s, vt
        self.safe.eval(result)
        return result

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
            biases = None
            self.safe.eval(weight_q, scales)
            return weight_q, scales, biases
        weight_q, scales, biases = result
        self.safe.eval(weight_q, scales, biases)
        return weight_q, scales, biases

    def dequantize(
        self,
        weight: Array,
        scales: Array,
        biases: Array | None,
        group_size: int,
        bits: int,
        mode: str,
    ) -> Array:
        arr = self.mx.dequantize(
            weight,
            scales=scales,
            biases=biases,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        self.safe.eval(arr)
        return arr

    def eval(self, *arrays: Array) -> None:
        self.safe.eval(*arrays)

    def to_numpy(self, array: Array) -> Any:
        self.safe.eval(array)
        return np.array(array)

    # --- Array Creation (new) ---
    def eye(self, n: int, m: int | None = None, dtype: Any | None = None) -> Array:
        arr = self.mx.eye(n, m, dtype=self._map_dtype(dtype))
        self.safe.eval(arr)
        return arr

    def arange(
        self,
        start: int | float,
        stop: int | float | None = None,
        step: int | float = 1,
        dtype: Any | None = None,
    ) -> Array:
        if stop is None:
            arr = self.mx.arange(start, dtype=self._map_dtype(dtype))
        else:
            arr = self.mx.arange(start, stop, step, dtype=self._map_dtype(dtype))
        self.safe.eval(arr)
        return arr

    def diag(self, array: Array, k: int = 0) -> Array:
        arr = self.mx.diag(array, k=k)
        self.safe.eval(arr)
        return arr

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: Any | None = None) -> Array:
        arr = self.mx.full(shape, fill_value, dtype=self._map_dtype(dtype))
        self.safe.eval(arr)
        return arr

    def ones_like(self, array: Array, dtype: Any | None = None) -> Array:
        arr = self.mx.ones_like(array, dtype=self._map_dtype(dtype)) if dtype else self.mx.ones_like(array)
        self.safe.eval(arr)
        return arr

    def zeros_like(self, array: Array, dtype: Any | None = None) -> Array:
        arr = self.mx.zeros_like(array, dtype=self._map_dtype(dtype)) if dtype else self.mx.zeros_like(array)
        self.safe.eval(arr)
        return arr

    def linspace(self, start: float, stop: float, num: int, dtype: Any | None = None) -> Array:
        arr = self.mx.linspace(start, stop, num, dtype=self._map_dtype(dtype))
        self.safe.eval(arr)
        return arr

    # --- Shape Manipulation (new) ---
    def stack(self, arrays: list[Array], axis: int = 0) -> Array:
        arr = self.mx.stack(arrays, axis=axis)
        self.safe.eval(arr)
        return arr

    def concatenate(self, arrays: list[Array], axis: int = 0) -> Array:
        arr = self.mx.concatenate(arrays, axis=axis)
        self.safe.eval(arr)
        return arr

    def broadcast_to(self, array: Array, shape: tuple[int, ...]) -> Array:
        arr = self.mx.broadcast_to(array, shape)
        self.safe.eval(arr)
        return arr

    # --- Reductions (new) ---
    def mean(self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        arr = self.mx.mean(array, axis=axis, keepdims=keepdims)
        self.safe.eval(arr)
        return arr

    def min(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        arr = self.mx.min(array, axis=axis, keepdims=keepdims)
        self.safe.eval(arr)
        return arr

    def argmax(self, array: Array, axis: int | None = None) -> Array:
        arr = self.mx.argmax(array, axis=axis)
        self.safe.eval(arr)
        return arr

    def argmin(self, array: Array, axis: int | None = None) -> Array:
        arr = self.mx.argmin(array, axis=axis)
        self.safe.eval(arr)
        return arr

    def var(self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        arr = self.mx.var(array, axis=axis, keepdims=keepdims)
        self.safe.eval(arr)
        return arr

    def std(self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        arr = self.mx.std(array, axis=axis, keepdims=keepdims)
        self.safe.eval(arr)
        return arr

    # --- Element-wise Operations (new) ---
    def sign(self, array: Array) -> Array:
        arr = self.mx.sign(array)
        self.safe.eval(arr)
        return arr

    def clip(self, array: Array, min_val: float | Array | None, max_val: float | Array | None) -> Array:
        arr = self.mx.clip(array, min_val, max_val)
        self.safe.eval(arr)
        return arr

    def where(self, condition: Array, x: Array, y: Array) -> Array:
        arr = self.mx.where(condition, x, y)
        self.safe.eval(arr)
        return arr

    def softmax(self, array: Array, axis: int = -1) -> Array:
        arr = self.mx.softmax(array, axis=axis)
        self.safe.eval(arr)
        return arr

    def cumsum(self, array: Array, axis: int | None = None) -> Array:
        arr = self.mx.cumsum(array, axis=axis)
        self.safe.eval(arr)
        return arr

    # --- Linear Algebra (new) ---
    def dot(self, a: Array, b: Array) -> Array:
        # MLX uses matmul for general case; for 1D vectors use sum of element-wise product
        if a.ndim == 1 and b.ndim == 1:
            arr = self.mx.sum(a * b)
        else:
            arr = self.mx.matmul(a, b)
        self.safe.eval(arr)
        return arr

    def norm(self, array: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        arr = self.mx.linalg.norm(array, axis=axis, keepdims=keepdims)
        self.safe.eval(arr)
        return arr

    def det(self, array: Array) -> Array:
        arr = self.mx.linalg.det(array)
        self.safe.eval(arr)
        return arr

    def eigh(self, array: Array) -> tuple[Array, Array]:
        eigenvalues, eigenvectors = self.mx.linalg.eigh(array)
        self.safe.eval(eigenvalues, eigenvectors)
        return eigenvalues, eigenvectors

    def solve(self, a: Array, b: Array) -> Array:
        arr = self.mx.linalg.solve(a, b)
        self.safe.eval(arr)
        return arr

    def qr(self, array: Array) -> tuple[Array, Array]:
        q, r = self.mx.linalg.qr(array)
        self.safe.eval(q, r)
        return q, r

    # --- Sorting (new) ---
    def sort(self, array: Array, axis: int = -1) -> Array:
        arr = self.mx.sort(array, axis=axis)
        self.safe.eval(arr)
        return arr

    def argsort(self, array: Array, axis: int = -1) -> Array:
        arr = self.mx.argsort(array, axis=axis)
        self.safe.eval(arr)
        return arr

    # --- Random (new) ---
    def random_normal(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        arr = self.mx.random.normal(shape=shape, dtype=self._map_dtype(dtype) or self.mx.float32)
        self.safe.eval(arr)
        return arr

    def random_uniform(
        self,
        low: float = 0.0,
        high: float = 1.0,
        shape: tuple[int, ...] | None = None,
        dtype: Any | None = None,
    ) -> Array:
        arr = self.mx.random.uniform(
            low=low,
            high=high,
            shape=shape or (1,),
            dtype=self._map_dtype(dtype) or self.mx.float32,
        )
        self.safe.eval(arr)
        return arr

    def random_randint(self, low: int, high: int, shape: tuple[int, ...] | None = None) -> Array:
        arr = self.mx.random.randint(low, high, shape=shape or (1,))
        self.safe.eval(arr)
        return arr

    def random_seed(self, seed: int) -> None:
        self.mx.random.seed(seed)

    def _map_dtype(self, dtype: Any | None) -> Any | None:
        if dtype is None:
            return None
        if dtype is np.float32:
            return self.mx.float32
        if dtype is np.float16:
            return self.mx.float16
        if dtype is np.int32:
            return self.mx.int32
        if dtype is np.int64:
            return self.mx.int64
        return dtype
