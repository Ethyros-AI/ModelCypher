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

"""Tests for adapter routing domain dataclasses."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from modelcypher.core.domain.inference.adapter_routing import (
    AdapterIdentity,
    AdapterPool,
    LayerRoutingMeasurement,
    LayerRoutingSnapshot,
    RoutingTrace,
)


def test_adapter_identity_is_frozen() -> None:
    identity = AdapterIdentity(
        id="math",
        path="/tmp/math",
        rank=16,
        scale=1.0,
        module_keys=("layer.0.q_proj",),
    )

    with pytest.raises(FrozenInstanceError):
        identity.rank = 8  # type: ignore[misc]


def test_layer_measurement_and_snapshot_is_frozen() -> None:
    measurement = LayerRoutingMeasurement(
        layer_index=3,
        adapter_id="math",
        kl_divergence=0.12,
        activation_norm_ratio=1.01,
        cosine_similarity=0.98,
    )
    snapshot = LayerRoutingSnapshot(
        layer_index=3,
        measurements=(measurement,),
        base_activation_norm=7.5,
    )

    with pytest.raises(FrozenInstanceError):
        snapshot.layer_index = 2  # type: ignore[misc]


def test_routing_trace_is_frozen() -> None:
    identity = AdapterIdentity(
        id="math",
        path="/tmp/math",
        rank=16,
        scale=1.0,
        module_keys=("layer.0.q_proj",),
    )
    measurement = LayerRoutingMeasurement(
        layer_index=0,
        adapter_id="math",
        kl_divergence=0.02,
        activation_norm_ratio=1.0,
        cosine_similarity=1.0,
    )
    snapshot = LayerRoutingSnapshot(
        layer_index=0,
        measurements=(measurement,),
        base_activation_norm=3.0,
    )
    trace = RoutingTrace(
        prompt="What is 2+2?",
        n_adapters=1,
        n_layers=1,
        layer_snapshots=(snapshot,),
        adapter_identities=(identity,),
        selected_adapter_per_layer={0: "math"},
        selection_method="min_kl",
        output_kl_vs_single=None,
    )

    with pytest.raises(FrozenInstanceError):
        trace.prompt = "Changed"  # type: ignore[misc]


def test_adapter_pool_is_mutable() -> None:
    pool = AdapterPool(
        base_model=object(),
        base_tokenizer=object(),
        base_model_path="/tmp/base",
        adapter_models={},
        adapter_identities=(),
    )

    pool.base_model_path = "/tmp/new-base"
    assert pool.base_model_path == "/tmp/new-base"

