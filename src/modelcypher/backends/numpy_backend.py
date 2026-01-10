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

"""NumPy backend for CPU fallback and test environments.

This backend is intended for:
- sandboxed environments where GPU backends cannot initialize
- lightweight smoke tests and CI

Domain code still uses the Backend protocol; this implementation simply
executes those operations eagerly on CPU.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from safetensors.numpy import load_file, save_file

from modelcypher.ports.backend import Array, Backend, FloatInfo


class NumpyBackend(Backend):
    """CPU backend implemented with NumPy arrays."""

    def __init__(self) -> None:
        # Match MLX/JAX default float precision (float32) unless explicitly requested.
        self._default_float = np.float32

    # --- Dtype helpers ---
    def _map_dtype(self, dtype: Any | None) -> Any | None:
        if dtype is None:
            return None
        if isinstance(dtype, str):
            normalized = dtype.replace("numpy.", "").replace("np.", "")
            if normalized.startswith("mlx.core."):
                normalized = normalized.replace("mlx.core.", "")
            if normalized.startswith("jax.numpy."):
                normalized = normalized.replace("jax.numpy.", "")
            dtype_map: dict[str, Any] = {
                "float16": np.float16,
                "float32": np.float32,
                "float64": np.float64,
                "bfloat16": getattr(np, "bfloat16", np.float32),
                "int8": np.int8,
                "int16": np.int16,
                "int32": np.int32,
                "int64": np.int64,
                "uint8": np.uint8,
                "bool": np.bool_,
            }
            return dtype_map.get(normalized, dtype)

        name = getattr(dtype, "name", None) or getattr(dtype, "__name__", None) or str(dtype)
        return self._map_dtype(str(name))

    def finfo(self, dtype: Any | None = None) -> FloatInfo:
        np_dtype = self._map_dtype(dtype) or self._default_float
        try:
            info = np.finfo(np_dtype)
        except Exception:
            info = np.finfo(self._default_float)
        return FloatInfo(
            eps=float(info.eps),
            tiny=float(info.tiny),
            max=float(info.max),
            min=float(info.min),
        )

    def dtype(self, array: Array) -> Any:
        return getattr(array, "dtype", None)

    # --- Array Creation ---
    def array(self, data: Any, dtype: Any | None = None) -> Array:
        mapped_dtype = self._map_dtype(dtype)
        if isinstance(data, np.ndarray):
            if mapped_dtype is None:
                return data.astype(self._default_float, copy=False) if data.dtype == np.float64 else data
            return data.astype(mapped_dtype, copy=False)

        arr = np.asarray(data, dtype=mapped_dtype)
        if mapped_dtype is None and getattr(arr, "dtype", None) == np.float64:
            arr = arr.astype(self._default_float, copy=False)
        return arr

    def zeros(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        return np.zeros(shape, dtype=self._map_dtype(dtype) or self._default_float)

    def ones(self, shape: tuple[int, ...], dtype: Any | None = None) -> Array:
        return np.ones(shape, dtype=self._map_dtype(dtype) or self._default_float)

    def eye(self, n: int, m: int | None = None, dtype: Any | None = None) -> Array:
        return np.eye(n, M=m, dtype=self._map_dtype(dtype) or self._default_float)

    def arange(
        self,
        start: int | float,
        stop: int | float | None = None,
        step: int | float = 1,
        dtype: Any | None = None,
    ) -> Array:
        mapped_dtype = self._map_dtype(dtype)
        if stop is None:
            return np.arange(start, dtype=mapped_dtype)
        return np.arange(start, stop, step, dtype=mapped_dtype)

    def triu_indices(self, n: int, k: int = 0) -> tuple[Array, Array]:
        return np.triu_indices(n, k=k)

    def diag(self, array: Array, k: int = 0) -> Array:
        return np.diag(array, k=k)

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: Any | None = None) -> Array:
        return np.full(shape, fill_value, dtype=self._map_dtype(dtype) or self._default_float)

    def ones_like(self, array: Array, dtype: Any | None = None) -> Array:
        return np.ones_like(array, dtype=self._map_dtype(dtype))

    def zeros_like(self, array: Array, dtype: Any | None = None) -> Array:
        return np.zeros_like(array, dtype=self._map_dtype(dtype))

    def linspace(self, start: float, stop: float, num: int, dtype: Any | None = None) -> Array:
        return np.linspace(start, stop, num, dtype=self._map_dtype(dtype) or self._default_float)

    def meshgrid(self, *arrays: Array, indexing: str = "xy") -> list[Array]:
        return list(np.meshgrid(*arrays, indexing=indexing))

    # --- Shape Manipulation ---
    def shape(self, array: Array) -> tuple[int, ...]:
        return tuple(np.shape(array))

    def reshape(self, array: Array, shape: tuple[int, ...]) -> Array:
        return np.reshape(array, shape)

    def squeeze(self, array: Array, axis: int | None = None) -> Array:
        return np.squeeze(array, axis=axis)

    def transpose(self, array: Array, axes: tuple[int, ...] | None = None) -> Array:
        return np.transpose(array, axes=axes) if axes is not None else np.transpose(array)

    def stack(self, arrays: list[Array], axis: int = 0) -> Array:
        return np.stack(arrays, axis=axis)

    def concatenate(self, arrays: list[Array], axis: int = 0) -> Array:
        return np.concatenate(arrays, axis=axis)

    def broadcast_to(self, array: Array, shape: tuple[int, ...]) -> Array:
        return np.broadcast_to(array, shape)

    def tile(self, array: Array, reps: tuple[int, ...] | int) -> Array:
        return np.tile(array, reps)

    # --- Core Ops ---
    def matmul(self, lhs: Array, rhs: Array) -> Array:
        return np.matmul(lhs, rhs)

    def sum(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return np.sum(array, axis=axis, keepdims=keepdims)

    def mean(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return np.mean(array, axis=axis, keepdims=keepdims)

    def max(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return np.max(array, axis=axis, keepdims=keepdims)

    def min(self, array: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return np.min(array, axis=axis, keepdims=keepdims)

    def sqrt(self, array: Array) -> Array:
        return np.sqrt(array)

    def exp(self, array: Array) -> Array:
        return np.exp(array)

    def log(self, array: Array) -> Array:
        return np.log(array)

    def maximum(self, lhs: Array, rhs: Array) -> Array:
        return np.maximum(lhs, rhs)

    def minimum(self, lhs: Array, rhs: Array) -> Array:
        return np.minimum(lhs, rhs)

    def abs(self, array: Array) -> Array:
        return np.abs(array)

    def astype(self, array: Array, dtype: Any) -> Array:
        return np.asarray(array).astype(self._map_dtype(dtype), copy=False)

    def clip(self, array: Array, min_value: float, max_value: float) -> Array:
        return np.clip(array, min_value, max_value)

    def where(self, condition: Array, x: Array, y: Array) -> Array:
        return np.where(condition, x, y)

    def sign(self, array: Array) -> Array:
        return np.sign(array)

    def isnan(self, array: Array) -> Array:
        return np.isnan(array)

    def isinf(self, array: Array) -> Array:
        return np.isinf(array)

    def isfinite(self, array: Array) -> Array:
        return np.isfinite(array)

    # --- Indexing ---
    def take(self, array: Array, indices: Array, axis: int | None = None) -> Array:
        return np.take(array, indices, axis=axis)

    def take_along_axis(self, array: Array, indices: Array, axis: int) -> Array:
        return np.take_along_axis(array, indices, axis=axis)

    def put_along_axis(
        self, array: Array, indices: Array, values: Array, axis: int | None = None
    ) -> Array:
        out = np.array(array, copy=True)
        if axis is None:
            flat = out.reshape(-1)
            flat_idx = np.asarray(indices).reshape(-1)
            flat_vals = np.asarray(values).reshape(-1)
            flat[flat_idx] = flat_vals
            return out
        np.put_along_axis(out, indices, values, axis=axis)
        return out

    def nonzero(self, array: Array) -> tuple[Array, ...]:
        return np.nonzero(array)

    # --- Sorting ---
    def sort(self, array: Array, axis: int = -1) -> Array:
        return np.sort(array, axis=axis)

    def argsort(self, array: Array, axis: int = -1) -> Array:
        return np.argsort(array, axis=axis)

    def argpartition(self, array: Array, kth: int, axis: int = -1) -> Array:
        return np.argpartition(array, kth, axis=axis)

    def partition(self, array: Array, kth: int, axis: int = -1) -> Array:
        return np.partition(array, kth, axis=axis)

    # --- Graph Algorithms ---
    def floyd_warshall(self, dist: Array) -> Array:
        dist_arr = np.asarray(dist)
        if dist_arr.ndim != 2 or dist_arr.shape[0] != dist_arr.shape[1]:
            raise ValueError("floyd_warshall requires a square [n, n] matrix")
        out = dist_arr.copy()
        n = int(out.shape[0])
        for k in range(n):
            via = out[:, k : k + 1] + out[k : k + 1, :]
            out = np.minimum(out, via)
        return out

    # --- Extraction ---
    def tolist(self, array: Array) -> Any:
        return np.asarray(array).tolist()

    def to_numpy(self, array: Array) -> Any:
        return np.asarray(array)

    def to_scalar(self, array: Array) -> float:
        value = np.asarray(array)
        return float(value.item()) if value.shape == () else float(value.reshape(-1)[0].item())

    # --- Compute Control ---
    def eval(self, *arrays: Array) -> None:
        return None

    def clear_cache(self) -> None:
        return None

    # --- Performance APIs (no-op on NumPy) ---
    def compile(
        self,
        fun: Callable,
        inputs: list | None = None,
        outputs: list | None = None,
        shapeless: bool = False,
    ) -> Callable:
        return fun

    # --- Stream Management (no-op on NumPy) ---
    def new_stream(self, device: str = "gpu") -> Any:
        return None

    def synchronize(self) -> None:
        return None

    # --- File I/O ---
    def save_safetensors(
        self, path: str, weights: dict[str, Array], metadata: dict[str, str] | None = None
    ) -> None:
        np_weights = {key: np.asarray(value) for key, value in weights.items()}
        save_file(np_weights, path, metadata=metadata or {})

    def load_safetensors(self, path: str) -> dict[str, Array]:
        return load_file(path)

