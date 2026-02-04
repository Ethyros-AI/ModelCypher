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

"""Cross-architecture weight key mapping utilities."""

from __future__ import annotations

from typing import Callable

# Semantic weight name mappings (bidirectional)
_WEIGHT_NAME_EQUIVALENTS = [
    ("feed_forward.w1", "mlp.gate_proj"),
    ("feed_forward.w2", "mlp.down_proj"),
    ("feed_forward.w3", "mlp.up_proj"),
    ("self_attn.out_proj", "self_attn.o_proj"),
    ("operator_norm", "input_layernorm"),
    ("ffn_norm", "post_attention_layernorm"),
]


def _map_weight_key_cross_arch(
    target_key: str,
    source_keys: set[str],
    layer_mapping: dict[int, int] | None,
    extract_layer_fn: "Callable[[str], int | None]",
) -> str | None:
    """Map a target weight key to an equivalent source weight key."""
    if target_key in source_keys:
        return target_key

    target_layer = extract_layer_fn(target_key)
    if target_layer is None:
        return None

    source_layer = layer_mapping.get(target_layer, target_layer) if layer_mapping else target_layer

    for tgt_pattern, src_pattern in _WEIGHT_NAME_EQUIVALENTS:
        if tgt_pattern in target_key:
            candidate = target_key.replace(
                f"layers.{target_layer}",
                f"layers.{source_layer}",
            ).replace(tgt_pattern, src_pattern)
            if candidate in source_keys:
                return candidate

        if src_pattern in target_key:
            candidate = target_key.replace(
                f"layers.{target_layer}",
                f"layers.{source_layer}",
            ).replace(src_pattern, tgt_pattern)
            if candidate in source_keys:
                return candidate

    if layer_mapping and target_layer != source_layer:
        candidate = target_key.replace(
            f"layers.{target_layer}",
            f"layers.{source_layer}",
        )
        if candidate in source_keys:
            return candidate

    return None
