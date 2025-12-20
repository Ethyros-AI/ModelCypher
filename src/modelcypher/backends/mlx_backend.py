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

    def transpose(self, array: Array) -> Array:
        arr = self.mx.transpose(array)
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

    def quantize(
        self,
        array: Array,
        group_size: int,
        bits: int,
        mode: str,
    ) -> tuple[Array, Array, Array | None]:
        result = self.mx.quantize(
            array,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        if len(result) == 3:
            w_q, scales, biases = result
        else:
            w_q, scales = result
            biases = None
        self.safe.eval(w_q, scales)
        if biases is not None:
            self.safe.eval(biases)
        return w_q, scales, biases

    def eval(self, *arrays: Array) -> None:
        self.safe.eval(*arrays)

    def to_numpy(self, array: Array) -> Any:
        self.safe.eval(array)
        return np.array(array)

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
