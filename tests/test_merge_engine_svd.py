from __future__ import annotations

import numpy as np

from modelcypher.core.use_cases.merge_engine import RotationalMerger


class NumpyBackend:
    def __init__(self) -> None:
        self.matmul_calls = 0

    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype)

    def transpose(self, array):
        return np.transpose(array)

    def matmul(self, lhs, rhs):
        self.matmul_calls += 1
        return lhs @ rhs

    def eval(self, *arrays) -> None:
        return None

    def to_numpy(self, array):
        return np.array(array)


class MergerHarness(RotationalMerger):
    def __init__(self, backend) -> None:
        self.backend = backend


class TrackingMerger(MergerHarness):
    def __init__(self, backend, forbidden) -> None:
        super().__init__(backend)
        self._forbidden = forbidden
        self._to_numpy_calls = []

    def _to_numpy(self, value):
        if value is self._forbidden:
            raise AssertionError("weight converted to numpy in truncated SVD path")
        self._to_numpy_calls.append(value)
        return super()._to_numpy(value)


def test_truncated_svd_uses_backend_matmul_without_weight_numpy() -> None:
    backend = NumpyBackend()
    weight = np.arange(1, 13, dtype=np.float32).reshape(4, 3)
    merger = TrackingMerger(backend, forbidden=weight)

    bases = merger._truncated_svd_bases(
        weight=weight,
        rank=2,
        oversampling=1,
        power_iterations=1,
        seed=0,
        label="unit-test",
    )

    assert backend.matmul_calls > 0
    assert bases.u.shape == (4, 2)
    assert bases.v.shape == (3, 2)
