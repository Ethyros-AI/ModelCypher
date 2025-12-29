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

import pytest


def test_mlx_backend_array_accepts_uint32_numpy():
    mx = pytest.importorskip("mlx.core")

    import numpy as np

    from modelcypher.backends.mlx_backend import MLXBackend

    backend = MLXBackend()

    # Regression: converting uint32 via Python ints can raise MLX std::bad_cast.
    value = np.array([2**32 - 1], dtype=np.uint32)
    arr = backend.array(value)

    assert arr.shape == (1,)
    assert arr.dtype == mx.uint32

