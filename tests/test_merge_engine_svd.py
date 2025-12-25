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

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.use_cases.merge_engine import RotationalMerger
from tests.conftest import HAS_MLX

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


class TrackingBackend:
    """Wrapper around MLXBackend that tracks matmul calls."""

    def __init__(self) -> None:
        from modelcypher.backends.mlx_backend import MLXBackend
        self._backend = MLXBackend()
        self.matmul_calls = 0

    def array(self, data, dtype=None):
        return self._backend.array(data, dtype=dtype)

    def transpose(self, array, axes=None):
        return self._backend.transpose(array, axes=axes)

    def matmul(self, lhs, rhs):
        self.matmul_calls += 1
        return self._backend.matmul(lhs, rhs)

    def eval(self, *arrays) -> None:
        return self._backend.eval(*arrays)

    def to_numpy(self, array):
        return self._backend.to_numpy(array)


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
    backend = TrackingBackend()
    default_backend = get_default_backend()
    weight_data = list(range(1, 13))
    weight = default_backend.array(weight_data, dtype=None)
    weight = default_backend.reshape(weight, (4, 3))
    weight = default_backend.to_numpy(weight).astype('float32')  # Convert to numpy for test comparison
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
