from __future__ import annotations

import numpy as np

from modelcypher.ports.backend import Backend


class NumpyBackend(Backend):
    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype)

    def reshape(self, array, shape):
        return np.reshape(array, shape)

    def squeeze(self, array, axis=None):
        return np.squeeze(array, axis=axis)

    def transpose(self, array):
        return np.transpose(array)

    def matmul(self, lhs, rhs):
        return lhs @ rhs

    def sum(self, array, axis=None, keepdims=False):
        return np.sum(array, axis=axis, keepdims=keepdims)

    def max(self, array, axis=None, keepdims=False):
        return np.max(array, axis=axis, keepdims=keepdims)

    def sqrt(self, array):
        return np.sqrt(array)

    def exp(self, array):
        return np.exp(array)

    def log(self, array):
        return np.log(array)

    def maximum(self, lhs, rhs):
        return np.maximum(lhs, rhs)

    def minimum(self, lhs, rhs):
        return np.minimum(lhs, rhs)

    def abs(self, array):
        return np.abs(array)

    def astype(self, array, dtype):
        return array.astype(dtype)

    def svd(self, array, compute_uv=True):
        if compute_uv:
            u, s, vt = np.linalg.svd(array, full_matrices=False)
            return u, s, vt
        return np.linalg.svd(array, full_matrices=False, compute_uv=False)

    def eval(self, *arrays):
        return None

    def to_numpy(self, array):
        return np.array(array)


__all__ = ["NumpyBackend"]
