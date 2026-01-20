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


def test_jax_attention_sinks_bias_output():
    """JAX backend should apply sinks as additive bias before softmax."""
    pytest.importorskip("jax")

    from modelcypher.backends.jax_backend import JAXBackend

    backend = JAXBackend()
    q = backend.array([[[[1.0]], [[1.0]]]])
    k = backend.array([[[[1.0]], [[1.0]]]])
    v = backend.array([[[[1.0]], [[3.0]]]])

    out_no_sinks = backend.scaled_dot_product_attention(q, k, v, scale=1.0)
    out_with_sinks = backend.scaled_dot_product_attention(
        q,
        k,
        v,
        scale=1.0,
        sinks=[[0.0, 1.0], [0.0, 1.0]],
    )

    out_no_val = backend.tolist(out_no_sinks)[0][0][0][0]
    out_sink_val = backend.tolist(out_with_sinks)[0][0][0][0]

    assert out_sink_val > out_no_val
