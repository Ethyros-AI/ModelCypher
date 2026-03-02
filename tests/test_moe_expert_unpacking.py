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

"""Tests for MoE expert unpacking correctness."""

from __future__ import annotations

import pytest

mlx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from modelcypher.backends.moe_expert_unpacking import (
    UnpackedExpert,
    UnpackedMoEBlock,
    _make_linear_from_weight,
    unpack_moe_block,
    unpack_model_experts,
)
from modelcypher.core.domain.moe.topology import MoETopology


def _make_packed_moe_block(
    num_experts: int = 4,
    hidden: int = 32,
    intermediate: int = 16,
    top_k: int = 2,
) -> tuple:
    """Create a minimal packed MoE block matching Qwen3NextSparseMoeBlock layout."""
    from types import SimpleNamespace

    from mlx_lm.models.switch_layers import SwitchGLU

    gate = nn.Linear(hidden, num_experts, bias=False)
    switch_mlp = SwitchGLU(hidden, intermediate, num_experts, bias=False)
    shared_expert_mlp = SimpleNamespace(
        gate_proj=nn.Linear(hidden, intermediate, bias=False),
        up_proj=nn.Linear(hidden, intermediate, bias=False),
        down_proj=nn.Linear(intermediate, hidden, bias=False),
    )

    def shared_expert_call(x):
        return shared_expert_mlp.down_proj(
            nn.silu(shared_expert_mlp.gate_proj(x)) * shared_expert_mlp.up_proj(x)
        )

    shared_expert_gate = nn.Linear(hidden, 1, bias=False)

    block = SimpleNamespace(
        gate=gate,
        switch_mlp=switch_mlp,
        shared_expert=SimpleNamespace(__call__=shared_expert_call),
        shared_expert_gate=shared_expert_gate,
        norm_topk_prob=True,
        top_k=top_k,
    )
    # Make shared_expert callable
    block.shared_expert = type(
        "SharedExpert", (), {"__call__": lambda self, x: shared_expert_call(x)}
    )()

    mlx.eval(gate.parameters(), switch_mlp.parameters(), shared_expert_gate.parameters())
    mlx.eval(
        shared_expert_mlp.gate_proj.parameters(),
        shared_expert_mlp.up_proj.parameters(),
        shared_expert_mlp.down_proj.parameters(),
    )

    return block, num_experts, top_k, hidden, intermediate


def test_make_linear_from_weight_preserves_shape():
    w = mlx.random.normal((16, 32))
    mlx.eval(w)
    linear = _make_linear_from_weight(w)
    assert linear.weight.shape == (16, 32)
    assert mlx.array_equal(linear.weight, w)


def test_unpacked_expert_forward_matches_swiglu():
    """Single expert forward: down(silu(gate(x)) * up(x))."""
    hidden, intermediate = 32, 16
    gate_proj = nn.Linear(hidden, intermediate, bias=False)
    up_proj = nn.Linear(hidden, intermediate, bias=False)
    down_proj = nn.Linear(intermediate, hidden, bias=False)
    mlx.eval(gate_proj.parameters(), up_proj.parameters(), down_proj.parameters())

    expert = UnpackedExpert(gate_proj, up_proj, down_proj)

    x = mlx.random.normal((2, 4, hidden))
    mlx.eval(x)

    result = expert(x)
    expected = down_proj(nn.silu(gate_proj(x)) * up_proj(x))
    mlx.eval(result, expected)

    diff = mlx.abs(result - expected).max().item()
    assert diff < 1e-6, f"Max diff: {diff}"


def test_unpack_moe_block_produces_correct_expert_count():
    block, num_experts, top_k, hidden, intermediate = _make_packed_moe_block()
    unpacked = unpack_moe_block(block, num_experts=num_experts, num_experts_per_tok=top_k)

    assert isinstance(unpacked, UnpackedMoEBlock)
    assert len(unpacked.experts) == num_experts
    assert unpacked.top_k == top_k
    assert unpacked.norm_topk_prob is True


def test_unpack_moe_block_expert_weights_match_packed():
    """Each unpacked expert's weights equal the corresponding packed slice."""
    block, num_experts, top_k, hidden, intermediate = _make_packed_moe_block()
    unpacked = unpack_moe_block(block, num_experts=num_experts, num_experts_per_tok=top_k)

    gate_packed = block.switch_mlp.gate_proj.weight
    up_packed = block.switch_mlp.up_proj.weight
    down_packed = block.switch_mlp.down_proj.weight

    for e in range(num_experts):
        expert = unpacked.experts[e]
        mlx.eval(expert.gate_proj.weight, expert.up_proj.weight, expert.down_proj.weight)
        mlx.eval(gate_packed[e], up_packed[e], down_packed[e])

        assert mlx.array_equal(expert.gate_proj.weight, gate_packed[e])
        assert mlx.array_equal(expert.up_proj.weight, up_packed[e])
        assert mlx.array_equal(expert.down_proj.weight, down_packed[e])


def test_unpack_model_experts_replaces_packed_blocks():
    """unpack_model_experts replaces packed blocks in-place."""
    from types import SimpleNamespace

    block, num_experts, top_k, hidden, intermediate = _make_packed_moe_block()

    # Create a minimal model structure
    dense_mlp = SimpleNamespace(up_proj=nn.Linear(hidden, intermediate, bias=False))
    layer_moe = SimpleNamespace(mlp=block)
    layer_dense = SimpleNamespace(mlp=dense_mlp)
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer_dense, layer_moe]))

    topology = MoETopology(
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=intermediate,
        has_shared_expert=True,
        shared_expert_intermediate_size=intermediate,
        moe_layer_indices=[1],
        num_layers=2,
    )

    count = unpack_model_experts(model, topology)

    assert count == 1
    assert isinstance(model.model.layers[1].mlp, UnpackedMoEBlock)
    # Dense layer unchanged
    assert not isinstance(model.model.layers[0].mlp, UnpackedMoEBlock)


def test_unpack_model_experts_skips_non_packed_layers():
    """Layers without switch_mlp are not modified."""
    from types import SimpleNamespace

    block, num_experts, top_k, hidden, intermediate = _make_packed_moe_block()
    # Remove switch_mlp to simulate non-packed
    dense_mlp = SimpleNamespace(up_proj=nn.Linear(hidden, intermediate, bias=False))
    layer = SimpleNamespace(mlp=dense_mlp)
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))

    topology = MoETopology(
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=intermediate,
        has_shared_expert=False,
        shared_expert_intermediate_size=None,
        moe_layer_indices=[0],
        num_layers=1,
    )

    count = unpack_model_experts(model, topology)
    assert count == 0
