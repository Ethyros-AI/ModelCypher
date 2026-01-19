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

"""Hypothesis property tests for adapter persistence round-trips."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile

from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.adapters.activation_store import NPZActivationStore
from modelcypher.adapters.bridge_store import SafetensorsBridgeStore
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.bridge.generator import BridgeGeneratorResult
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


@st.composite
def _activation_specs(draw):
    count = draw(st.integers(min_value=1, max_value=3))
    specs: list[tuple[str, int, int, list[float]]] = []
    for idx in range(count):
        rows = draw(st.integers(min_value=1, max_value=4))
        cols = draw(st.integers(min_value=1, max_value=4))
        values = draw(
            st.lists(
                st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False, width=32),
                min_size=rows * cols,
                max_size=rows * cols,
            )
        )
        specs.append((f"arr_{idx}", rows, cols, values))
    return specs


@settings(max_examples=10, deadline=None)
@given(specs=_activation_specs())
def test_npz_activation_store_roundtrip(specs) -> None:
    backend = get_default_backend()
    store = NPZActivationStore()

    arrays = {}
    for name, rows, cols, values in specs:
        arr = backend.array(values)
        arr = backend.reshape(arr, (rows, cols))
        backend.eval(arr)
        arrays[name] = arr

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "activations.npz"
        store.save_probe_activations(path, arrays, backend)
        loaded = store.load_probe_activations(path, backend)
    assert loaded is not None

    for name, original in arrays.items():
        loaded_arr = loaded[name]
        backend.eval(loaded_arr)
        diff = backend.abs(original - loaded_arr)
        max_diff = backend.max(diff)
        backend.eval(max_diff)
        eps = division_epsilon(backend, original)
        assert float(backend.to_scalar(max_diff)) <= eps


@settings(max_examples=10, deadline=None)
@given(
    source_dim=st.integers(min_value=1, max_value=4),
    target_dim=st.integers(min_value=1, max_value=4),
    scale_ratio=st.floats(
        min_value=0.10000000149011612,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
    cka=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=32),
    raw_cka=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=32),
    n_samples=st.integers(min_value=1, max_value=16),
)
def test_bridge_store_roundtrip(
    source_dim: int,
    target_dim: int,
    scale_ratio: float,
    cka: float,
    raw_cka: float,
    n_samples: int,
) -> None:
    backend = get_default_backend()
    store = SafetensorsBridgeStore()

    backend.random_seed(42)
    transform = backend.random_normal((source_dim, target_dim))
    transform_inv = backend.random_normal((target_dim, source_dim))
    backend.eval(transform, transform_inv)

    result = BridgeGeneratorResult(
        transform=transform,
        transform_inv=transform_inv,
        scale_ratio=scale_ratio,
        source_dim=source_dim,
        target_dim=target_dim,
        cka_achieved=cka,
        numerical_deviation=1.0 - cka,
        raw_cka=raw_cka,
        n_samples=n_samples,
        source_name="source",
        target_name="target",
        created_at=datetime.now(timezone.utc),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "bridge.safetensors"
        store.save(path, result, backend=backend)
        bridge = store.load(path, backend=backend)

    diff = backend.abs(bridge.transform - transform)
    diff_inv = backend.abs(bridge.transform_inv - transform_inv)
    max_diff = backend.max(diff)
    max_diff_inv = backend.max(diff_inv)
    backend.eval(max_diff, max_diff_inv)
    eps = division_epsilon(backend, transform)
    assert float(backend.to_scalar(max_diff)) <= eps
    assert float(backend.to_scalar(max_diff_inv)) <= eps
    assert bridge.scale_ratio == scale_ratio
    assert bridge.source_dim == source_dim
    assert bridge.target_dim == target_dim
    assert bridge.source_name == "source"
    assert bridge.target_name == "target"
