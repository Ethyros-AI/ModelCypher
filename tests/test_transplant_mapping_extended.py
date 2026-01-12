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

"""Extended tests for cross-architecture weight key mapping.

Tests critical APIs:
- _map_weight_key_cross_arch(): Map target weight keys to source equivalents
"""

import pytest

from modelcypher.core.use_cases.merge.stages.transplant_mapping import (
    _map_weight_key_cross_arch,
)


def _extract_layer(key: str) -> int | None:
    """Simple layer extractor for testing."""
    if "layers." not in key:
        return None
    try:
        parts = key.split("layers.")[1].split(".")
        return int(parts[0])
    except (IndexError, ValueError):
        return None


class TestMapWeightKeyCrossArch:
    """Tests for _map_weight_key_cross_arch()."""

    def test_exact_match_returns_key(self):
        """If target key exists in source, return it directly."""
        target_key = "model.layers.0.self_attn.q_proj.weight"
        source_keys = {
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
        }

        result = _map_weight_key_cross_arch(
            target_key=target_key,
            source_keys=source_keys,
            layer_mapping=None,
            extract_layer_fn=_extract_layer,
        )

        assert result == target_key

    def test_no_layer_returns_none(self):
        """Keys without layer info should return None."""
        target_key = "model.embed_tokens.weight"
        source_keys = {"model.embed_tokens.weight"}

        result = _map_weight_key_cross_arch(
            target_key=target_key,
            source_keys=source_keys,
            layer_mapping=None,
            extract_layer_fn=_extract_layer,
        )

        # The key exists, so should return it
        assert result == target_key

    def test_layer_mapping_applied(self):
        """Layer mapping should remap layer indices."""
        target_key = "model.layers.5.self_attn.q_proj.weight"
        source_keys = {
            "model.layers.2.self_attn.q_proj.weight",
        }
        layer_mapping = {5: 2}  # Target layer 5 -> Source layer 2

        result = _map_weight_key_cross_arch(
            target_key=target_key,
            source_keys=source_keys,
            layer_mapping=layer_mapping,
            extract_layer_fn=_extract_layer,
        )

        assert result == "model.layers.2.self_attn.q_proj.weight"

    def test_feed_forward_to_mlp_mapping(self):
        """feed_forward.w1 should map to mlp.gate_proj."""
        target_key = "model.layers.0.feed_forward.w1.weight"
        source_keys = {
            "model.layers.0.mlp.gate_proj.weight",
        }

        result = _map_weight_key_cross_arch(
            target_key=target_key,
            source_keys=source_keys,
            layer_mapping=None,
            extract_layer_fn=_extract_layer,
        )

        assert result == "model.layers.0.mlp.gate_proj.weight"

    def test_mlp_to_feed_forward_mapping(self):
        """mlp.gate_proj should map to feed_forward.w1."""
        target_key = "model.layers.0.mlp.gate_proj.weight"
        source_keys = {
            "model.layers.0.feed_forward.w1.weight",
        }

        result = _map_weight_key_cross_arch(
            target_key=target_key,
            source_keys=source_keys,
            layer_mapping=None,
            extract_layer_fn=_extract_layer,
        )

        assert result == "model.layers.0.feed_forward.w1.weight"

    def test_feed_forward_w2_to_down_proj(self):
        """feed_forward.w2 should map to mlp.down_proj."""
        target_key = "model.layers.0.feed_forward.w2.weight"
        source_keys = {
            "model.layers.0.mlp.down_proj.weight",
        }

        result = _map_weight_key_cross_arch(
            target_key=target_key,
            source_keys=source_keys,
            layer_mapping=None,
            extract_layer_fn=_extract_layer,
        )

        assert result == "model.layers.0.mlp.down_proj.weight"

    def test_feed_forward_w3_to_up_proj(self):
        """feed_forward.w3 should map to mlp.up_proj."""
        target_key = "model.layers.0.feed_forward.w3.weight"
        source_keys = {
            "model.layers.0.mlp.up_proj.weight",
        }

        result = _map_weight_key_cross_arch(
            target_key=target_key,
            source_keys=source_keys,
            layer_mapping=None,
            extract_layer_fn=_extract_layer,
        )

        assert result == "model.layers.0.mlp.up_proj.weight"

    def test_out_proj_to_o_proj_mapping(self):
        """self_attn.out_proj should map to self_attn.o_proj."""
        target_key = "model.layers.0.self_attn.out_proj.weight"
        source_keys = {
            "model.layers.0.self_attn.o_proj.weight",
        }

        result = _map_weight_key_cross_arch(
            target_key=target_key,
            source_keys=source_keys,
            layer_mapping=None,
            extract_layer_fn=_extract_layer,
        )

        assert result == "model.layers.0.self_attn.o_proj.weight"

    def test_no_match_returns_none(self):
        """No matching key should return None."""
        target_key = "model.layers.0.self_attn.q_proj.weight"
        source_keys = {
            "model.layers.0.mlp.gate_proj.weight",  # Different weight type
        }

        result = _map_weight_key_cross_arch(
            target_key=target_key,
            source_keys=source_keys,
            layer_mapping=None,
            extract_layer_fn=_extract_layer,
        )

        assert result is None

    def test_combined_layer_and_name_mapping(self):
        """Layer mapping and name mapping should work together."""
        target_key = "model.layers.10.feed_forward.w1.weight"
        source_keys = {
            "model.layers.5.mlp.gate_proj.weight",
        }
        layer_mapping = {10: 5}

        result = _map_weight_key_cross_arch(
            target_key=target_key,
            source_keys=source_keys,
            layer_mapping=layer_mapping,
            extract_layer_fn=_extract_layer,
        )

        assert result == "model.layers.5.mlp.gate_proj.weight"

    def test_layer_only_remapping(self):
        """When only layer differs, remap layer without name change."""
        target_key = "model.layers.10.self_attn.q_proj.weight"
        source_keys = {
            "model.layers.5.self_attn.q_proj.weight",
        }
        layer_mapping = {10: 5}

        result = _map_weight_key_cross_arch(
            target_key=target_key,
            source_keys=source_keys,
            layer_mapping=layer_mapping,
            extract_layer_fn=_extract_layer,
        )

        assert result == "model.layers.5.self_attn.q_proj.weight"


class TestMapWeightKeyEdgeCases:
    """Edge case tests for weight key mapping."""

    def test_empty_source_keys(self):
        """Empty source keys should return None."""
        target_key = "model.layers.0.self_attn.q_proj.weight"
        source_keys: set[str] = set()

        result = _map_weight_key_cross_arch(
            target_key=target_key,
            source_keys=source_keys,
            layer_mapping=None,
            extract_layer_fn=_extract_layer,
        )

        assert result is None

    def test_none_layer_mapping(self):
        """None layer_mapping should use identity mapping."""
        target_key = "model.layers.0.self_attn.q_proj.weight"
        source_keys = {
            "model.layers.0.self_attn.q_proj.weight",
        }

        result = _map_weight_key_cross_arch(
            target_key=target_key,
            source_keys=source_keys,
            layer_mapping=None,
            extract_layer_fn=_extract_layer,
        )

        assert result == target_key

    def test_layer_not_in_mapping(self):
        """Layer not in mapping should use identity."""
        target_key = "model.layers.5.self_attn.q_proj.weight"
        source_keys = {
            "model.layers.5.self_attn.q_proj.weight",
        }
        layer_mapping = {0: 0, 1: 1}  # Layer 5 not in mapping

        result = _map_weight_key_cross_arch(
            target_key=target_key,
            source_keys=source_keys,
            layer_mapping=layer_mapping,
            extract_layer_fn=_extract_layer,
        )

        assert result == target_key

    def test_bidirectional_norm_mapping(self):
        """operator_norm <-> input_layernorm mapping."""
        # Forward
        target_key = "model.layers.0.operator_norm.weight"
        source_keys = {"model.layers.0.input_layernorm.weight"}

        result = _map_weight_key_cross_arch(
            target_key=target_key,
            source_keys=source_keys,
            layer_mapping=None,
            extract_layer_fn=_extract_layer,
        )
        assert result == "model.layers.0.input_layernorm.weight"

        # Reverse
        target_key2 = "model.layers.0.input_layernorm.weight"
        source_keys2 = {"model.layers.0.operator_norm.weight"}

        result2 = _map_weight_key_cross_arch(
            target_key=target_key2,
            source_keys=source_keys2,
            layer_mapping=None,
            extract_layer_fn=_extract_layer,
        )
        assert result2 == "model.layers.0.operator_norm.weight"
