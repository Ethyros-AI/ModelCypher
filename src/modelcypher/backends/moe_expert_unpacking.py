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

"""Unpack packed MoE SwitchGLU expert tensors into individual nn.Linear modules.

Unpacking is a lossless identity operation: same weights, different layout.
After unpacking, all existing geometric tools (SVD, NB-LoRA injection,
spectral bounds) work unchanged because each expert is a standard nn.Linear.

The packed format uses mx.gather_mm for efficient batched expert selection.
The unpacked format uses a loop over selected experts — functionally
identical, not fused.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from modelcypher.core.domain.moe.topology import MoETopology

logger = logging.getLogger(__name__)


def _silu(x: mx.array) -> mx.array:
    return nn.silu(x)


class UnpackedExpert(nn.Module):
    """Individual SwiGLU expert with standard nn.Linear projections."""

    def __init__(
        self,
        gate_proj: nn.Linear,
        up_proj: nn.Linear,
        down_proj: nn.Linear,
    ) -> None:
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_silu(self.gate_proj(x)) * self.up_proj(x))


class UnpackedMoEBlock(nn.Module):
    """MoE block with individual expert modules for NB-LoRA training.

    Functionally identical to Qwen3NextSparseMoeBlock but with unpacked
    experts.  Routing logic preserved exactly from the original.  Individual
    experts are standard ``nn.Linear`` modules that NB-LoRA can target.
    """

    def __init__(
        self,
        gate: Any,
        experts: list[UnpackedExpert],
        shared_expert: Any,
        shared_expert_gate: Any,
        top_k: int,
        norm_topk_prob: bool,
    ) -> None:
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

    def __call__(self, x: mx.array) -> mx.array:
        # Router: exact same logic as Qwen3NextSparseMoeBlock
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)

        # Expert computation: compute each expert for all tokens,
        # multiply by routing score (0 for non-routed tokens).
        # Correct and differentiable. Cost is O(E * T * forward_expert)
        # vs packed O(k * T * forward_expert). For training with NB-LoRA,
        # backprop through frozen base + unfrozen LoRA dominates cost.
        output = mx.zeros_like(x)
        for expert_idx in range(len(self.experts)):
            expert_out = self.experts[expert_idx](x)
            mask = inds == expert_idx  # [..., k]
            expert_score = (scores * mask).sum(axis=-1, keepdims=True)  # [..., 1]
            output = output + expert_score * expert_out

        y = output

        # Shared expert: unchanged
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

        return y + shared_y


def _make_linear_from_weight(weight_2d: mx.array) -> nn.Linear:
    """Create nn.Linear with the given weight, no random initialisation.

    ``weight_2d`` shape is ``[output_dims, input_dims]`` — the standard
    ``nn.Linear`` weight layout.
    """
    out_dim, in_dim = weight_2d.shape
    linear = nn.Linear(in_dim, out_dim, bias=False)
    linear.weight = weight_2d
    return linear


def unpack_moe_block(
    packed_block: Any,
    num_experts: int,
    num_experts_per_tok: int,
) -> UnpackedMoEBlock:
    """Convert a packed SwitchGLU MoE block to individual expert modules.

    Lossless: same weights, different layout.

    Args:
        packed_block: ``Qwen3NextSparseMoeBlock`` or equivalent with
            ``.gate``, ``.switch_mlp``, ``.shared_expert``,
            ``.shared_expert_gate`` attributes.
        num_experts: Total expert count.
        num_experts_per_tok: Top-k experts per token.

    Returns:
        ``UnpackedMoEBlock`` with per-expert ``nn.Linear`` modules.
    """
    switch_mlp = packed_block.switch_mlp

    gate_packed = switch_mlp.gate_proj.weight   # [E, intermediate, hidden]
    up_packed = switch_mlp.up_proj.weight        # [E, intermediate, hidden]
    down_packed = switch_mlp.down_proj.weight     # [E, hidden, intermediate]

    experts: list[UnpackedExpert] = []
    for e in range(num_experts):
        gate_linear = _make_linear_from_weight(gate_packed[e])
        up_linear = _make_linear_from_weight(up_packed[e])
        down_linear = _make_linear_from_weight(down_packed[e])
        experts.append(UnpackedExpert(gate_linear, up_linear, down_linear))

    norm_topk_prob = getattr(packed_block, "norm_topk_prob", True)

    return UnpackedMoEBlock(
        gate=packed_block.gate,
        experts=experts,
        shared_expert=packed_block.shared_expert,
        shared_expert_gate=packed_block.shared_expert_gate,
        top_k=num_experts_per_tok,
        norm_topk_prob=norm_topk_prob,
    )


def unpack_model_experts(model: Any, topology: "MoETopology") -> int:
    """Replace all packed MoE blocks in *model* with unpacked versions.

    Mutates model in-place.  Returns the number of layers unpacked.

    Args:
        model: Loaded MLX model (may have ``model`` attribute wrapping layers).
        topology: MoE topology from config.

    Returns:
        Number of MoE layers that were unpacked.
    """
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        return 0
    count = 0

    for layer_idx in topology.moe_layer_indices:
        layer = layers[layer_idx]
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        if not hasattr(mlp, "switch_mlp"):
            continue

        unpacked = unpack_moe_block(
            mlp,
            num_experts=topology.num_experts,
            num_experts_per_tok=topology.num_experts_per_tok,
        )
        layer.mlp = unpacked
        count += 1

    if count > 0:
        logger.info(
            "Unpacked %d packed MoE layers for per-expert training", count,
        )
    return count
